import os
import json
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time
import openai
import traceback


import asyncio

load_dotenv()

class EditingOperation(BaseModel):
    operation: str
    params: Optional[Dict[str, Any]] = Field(default=None)
    description: str

class EditingPlan(BaseModel):
    steps: List[EditingOperation]
    summary: str

class PlanningGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.name = "PlanningGenerator"
        self.description = "Generates high-level video editing plans"
        self.requires_library = ["openai"]
        self.requires_env = ["OPENAI_API_KEY"]
        self.streamable = True
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
        self.context_window = 10000
        self.state = "AWAITING_INSTRUCTION"

    async def generate_stream(self, queries: List[str], context: List[str], conversation: Dict = None):
        try:    
            async for result in self.generate(queries, context, conversation):
                yield result
        except Exception as e:
            logging.error(f"Error in PlanningGenerator: {str(e)}")
            yield {
                "message": json.dumps({"error": f"Error generating editing plan: {str(e)}"}),
                "finish_reason": "error"
            }

    async def generate(self, queries: List[str], context: List[str], conversation: Dict = None):
        try:
            import openai
            logging.info(f"PlanningGenerator.generate called with queries: {queries}")

            openai.api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL", "")
            if base_url:
                openai.api_base = base_url

            if "OPENAI_API_TYPE" in os.environ:
                openai.api_type = os.getenv("OPENAI_API_TYPE")
            if "OPENAI_API_BASE" in os.environ:
                openai.api_base = os.getenv("OPENAI_API_BASE")
            if "OPENAI_API_VERSION" in os.environ:
                openai.api_version = os.getenv("OPENAI_API_VERSION")

            

            if self.state == "AWAITING_INSTRUCTION":
                completion = await self.handle_initial_request(queries, context, conversation)
            elif self.state == "AWAITING_CONFIRMATION":
                completion = await self.handle_confirmation(queries, context, conversation)
            else:
                yield {
                    "message": json.dumps({"error": f"Unknown state: {self.state}"}),
                    "finish_reason": "error"
                }
                return

            logging.info(f"Raw completion: {completion}")

            response_content = completion['choices'][0]['message']['content']
            logging.info(f"Response content: {response_content}")

            plan = json.loads(response_content)
            logging.info(f"Parsed plan: {plan}")
            
            if 'steps' not in plan or 'summary' not in plan:
                raise KeyError("Generated plan is missing required keys")

            if not isinstance(plan['steps'], list):
                raise TypeError("'steps' in the generated plan should be a list")

            # Add more detailed logging for the plan structure
            logging.info(f"Plan keys: {plan.keys()}")
            logging.info(f"Steps type: {type(plan['steps'])}")
            if isinstance(plan['steps'], list):
                logging.info(f"Number of steps: {len(plan['steps'])}")
                for i, step in enumerate(plan['steps']):
                    logging.info(f"Step {i} keys: {step.keys()}")
                    logging.info(f"Step {i} content: {step}")

            # Return the plan without further processing
            yield {
                "message": json.dumps(plan),
                "finish_reason": "success"
            }

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse the generated plan as JSON: {str(e)}")
            yield {
                "message": json.dumps({"error": "Failed to parse the generated plan"}),
                "finish_reason": "error"
            }
        except (KeyError, TypeError, IndexError) as e:
            logging.error(f"Error in plan structure: {str(e)}")
            if plan:
                logging.error(f"Plan content: {plan}")
            yield {
                "message": json.dumps({"error": f"Invalid plan structure: {str(e)}"}),
                "finish_reason": "error"
            }
        except Exception as e:
            logging.error(f"Error in PlanningGenerator: {str(e)}")
            logging.error(traceback.format_exc())
            yield {
                "message": json.dumps({"error": f"Error in PlanningGenerator: {str(e)}"}),
                "finish_reason": "error"
            }

    def process_plan(self, plan):
        processed_plan = {"steps": [], "summary": plan["summary"]}
        for step in plan["steps"]:
            processed_step = {
                "operation": step["operation"],
                "description": step["description"],
                "params": step.get("params", {})
            }
            processed_plan["steps"].append(processed_step)
        return processed_plan

    async def handle_initial_request(self, queries: List[str], context: List[str], conversation: Dict = None):
        system_prompt = """
        You are an AI video editing assistant. Your role is to:
        1. Understand the user's editing request
        2. Break it down into manageable tasks
        3. Identify potential issues or needed clarifications
        4. Coordinate with specialized components to execute the tasks
        5. Provide feedback and ask for clarification when needed
        
        Create a plan of action and engage with the user. Each step should have an operation, parameters, and a description.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": " ".join(queries)}
        ]

        if context:
            messages.append({"role": "user", "content": f"Context: {' '.join(context)}"})

        start_time = time.time()
        logging.info("Making API call to OpenAI...")

        chat_completion_arguments = {
            "model": self.model_name,
            "messages": messages,
            "functions": [{
                "name": "generate_editing_plan",
                "description": "Generate a high-level video editing plan",
                "parameters": EditingPlan.model_json_schema()
            }],
            "function_call": {"name": "generate_editing_plan"}
        }

        if openai.api_type == "azure":
            chat_completion_arguments["deployment_id"] = self.model_name

        try:
            completion = await asyncio.wait_for(
                openai.ChatCompletion.acreate(**chat_completion_arguments),
                timeout=30  # Set a timeout of 30 seconds
            )
        except asyncio.TimeoutError:
            logging.error("OpenAI API call timed out after 30 seconds")
            return {
                "message": json.dumps({"error": "API call timed out"}),
                "finish_reason": "error"
            }

        end_time = time.time()
        logging.info(f"OpenAI API call completed in {end_time - start_time:.2f} seconds")

        function_response = completion.choices[0].message.function_call.arguments
        logging.info(f"Raw function response: {function_response}")

        try:
            parsed_plan = EditingPlan.model_validate_json(function_response)
            plan_dict = parsed_plan.model_dump()
        except ValueError as ve:
            logging.error(f"Validation error: {ve}")
            # Attempt to parse the response manually
            raw_data = json.loads(function_response)
            plan_dict = {"steps": [], "summary": "Error in plan generation"}
            for step in raw_data.get("steps", []):
                operation = step.get("operation", "Unknown operation")
                params = step.get("params", {})
                description = step.get("description", "No description provided")
                plan_dict["steps"].append({"operation": operation, "params": params, "description": description})

        logging.info(f"Generated Editing Plan: {plan_dict}")

        if conversation is not None:
            conversation['last_plan'] = json.dumps(plan_dict)

        self.state = "AWAITING_CONFIRMATION"

        return {
            "message": json.dumps({
                "plan": plan_dict,
                "user_prompt": "I've created a plan based on your request. Please review it and let me know if you want to proceed or if you need any changes."
            }),
            "finish_reason": "stop"
        }

    async def handle_confirmation(self, queries: List[str], context: List[str], conversation: Dict = None):
        system_prompt = """
        You are an AI assistant helping with video editing. The user has been presented with a plan and is now responding.
        Your task is to determine if the user is:
        1. Confirming the plan (in which case, respond with 'CONFIRM')
        2. Requesting changes (in which case, respond with 'CHANGE')
        3. Asking for clarification (in which case, respond with 'CLARIFY')
        Respond with only one of these words based on the user's input.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": " ".join(queries)}
        ]

        chat_completion_arguments = {
            "model": self.model_name,
            "messages": messages,
        }

        completion = await openai.ChatCompletion.acreate(**chat_completion_arguments)
        response = completion.choices[0].message.content.strip().upper()

        if response == 'CONFIRM':
            if conversation and 'last_plan' in conversation:
                plan = EditingPlan.model_validate_json(conversation['last_plan'])
                results = await self.execute_plan(plan)
                self.state = "AWAITING_INSTRUCTION"
                return {
                    "message": json.dumps({
                        "execution_results": results,
                        "user_prompt": "The plan has been executed. Is there anything else you'd like me to do?"
                    }),
                    "finish_reason": "stop"
                }
            else:
                return {
                    "message": json.dumps({
                        "error": "No plan found to execute. Please generate a plan first.",
                        "user_prompt": "I couldn't find a plan to execute. Could you please restate your original request?"
                    }),
                    "finish_reason": "error"
                }
        elif response == 'CHANGE':
            self.state = "AWAITING_INSTRUCTION"
            return {
                "message": json.dumps({
                    "user_prompt": "I understand you want to make changes. Could you please describe what changes you'd like to make to the plan?"
                }),
                "finish_reason": "stop"
            }
        elif response == 'CLARIFY':
            return {
                "message": json.dumps({
                    "user_prompt": "I'd be happy to clarify. What part of the plan would you like me to explain further?"
                }),
                "finish_reason": "stop"
            }
        else:
            return {
                "message": json.dumps({
                    "error": f"Unexpected response: {response}",
                    "user_prompt": "I'm sorry, I didn't understand your response. Could you please clarify if you want to proceed with the plan, make changes, or if you need any part of it explained?"
                }),
                "finish_reason": "error"
            }

    async def execute_plan(self, plan: EditingPlan):
        results = []
        for step in plan.steps:
            try:
                if step.operation == "insert_clip":
                    result = await self.insert_clip(step.params)
                elif step.operation == "delete_section":
                    result = await self.delete_section(step.params)
                else:
                    result = f"Operation {step.operation} not implemented"
                
                results.append({
                    "operation": step.operation,
                    "description": step.description,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "operation": step.operation,
                    "description": step.description,
                    "error": str(e),
                    "status": "failed"
                })
        return results

    async def insert_clip(self, params: Dict[str, Any]):
        # Placeholder for inserting a clip
        # This would interact with the VideoEditingGenerator
        return f"Inserted clip with parameters: {params}"

    async def delete_section(self, params: Dict[str, Any]):
        # Placeholder for deleting a section
        # This would interact with the VideoEditingGenerator
        return f"Deleted section with parameters: {params}"

    # Add more methods for other operations as needed