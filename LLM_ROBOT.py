import os
import sys
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from call_function import call_function, available_functions
from Robot_Tools.Robot_Motion_Tools import device_close
from Robot_Tools.Robot_Motion_Tools import move_to_home

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    MODEL_ID = "gemini-2.5-flash"
    verbose = False
    max_iter = 20

    # Optional initial prompt from CLI
    initial_prompt = None
    if len(sys.argv) >= 2:
        initial_prompt = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == "--verbose":
        verbose = True


    print("HOMING ROBOT")
    move_to_home()

    # TODO(student):
    # Write a system prompt that tells the LLM:
    # 1) it controls a Dobot robot,
    # 2) it should use tools instead of pretending actions happened,
    # 3) when a pick-and-place request is incomplete, it should ask for missing info,
    # 4) after capturing the scene, it should summarize detected blocks clearly.
    system_prompt = """
    You are an AI assistant that directly controls a physical Dobot robotic arm.
    You have access to tools that move the robot, capture the scene, detect objects, and perform pick-and-place operations.

    Rules you must always follow:
    1. NEVER simulate or describe a robot action without calling the appropriate tool. Every physical action (move, pick, place, home, capture, etc.) MUST be executed through a tool call.
    2. If a user asks you to pick and place an object but has not provided all necessary information (such as target object, destination coordinates, or block identity), ask the user for the missing details before calling any tool.
    3. After capturing the scene and detecting blocks, always summarize your findings clearly to the user — including each detected block's color, position, or any other relevant attributes returned by the tool.
    4. Be concise and factual. Do not speculate about robot state; use tools to find out.
    """

    print("\n================ SYSTEM PROMPT ================\n")
    print(system_prompt.strip(), "\n")

    # Conversation history (user + assistant + tool messages)
    messages = []

    # If we got an initial CLI prompt, use it as the first user message
    if initial_prompt:
        print("\n================ USER PROMPT (CLI) ================\n")
        print(initial_prompt)
        messages.append(
            types.Content(role="user", parts=[types.Part(text=initial_prompt)])
        )
    else:
        # Otherwise, ask interactively for the first input
        user_text = input("\nYou (type 'quit' to exit): ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            return
        messages.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

    config = types.GenerateContentConfig(
        tools=[available_functions],
        system_instruction=system_prompt
    )

    func_count = 0

    # ================= INTERACTIVE CONVERSATION LOOP =================
    while True:
        # For each user message, allow multiple tool/model turns
        for i in range(max_iter):
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=messages,
                config=config
            )

            # ------------ MODEL TEXT ------------
            if response.text:
                print("\n================ MODEL TEXT RESPONSE ================\n")
                print(response.text)

            if verbose and response.usage_metadata:
                print(f'prompt = {messages[-1].parts[0].text if messages else ""}')
                print(f'Response = {response.text}')
                print(f'Prompt Token = {response.usage_metadata.prompt_token_count}')
                print(f'Response Token = {response.usage_metadata.candidates_token_count}')

            # Add assistant content to history
            if response.candidates:
                for candidate in response.candidates:
                    if candidate and candidate.content:
                        messages.append(candidate.content)

            # ------------ TOOL CALLS ------------
            if response.function_calls:
                for function_call_part in response.function_calls:
                    func_count += 1
                    fname = getattr(function_call_part, "name", None)
                    fargs = getattr(function_call_part, "args", {})

                    print(f"\n================ FUNCTION CALL #{func_count} ================\n")
                    print(f"Function name: {fname}")
                    print("Arguments (tool prompt):")
                    try:
                        print(json.dumps(fargs, indent=2))
                    except TypeError:
                        print(fargs)

                    # Run tool
                    result = call_function(function_call_part, verbose=True)

                    print(f"\n================ FUNCTION RESULT #{func_count} ================\n")
                    print(result)

                    # Append tool result so the model can see it next iteration
                    messages.append(result)

                # continue inner for-loop to let the model react to the tool results
                continue

            # ---------- NO FUNCTION CALLS -> END OF THIS TURN ----------
            break  # break out of the max_iter loop; ready for next user input

        # ================= ASK FOR NEXT USER INPUT =================
        print("\n================ AWAITING USER INPUT (type 'quit' to exit) ================\n")
        user_text = input("You: ").strip()

        if user_text.lower() in {"quit", "exit", "q"}:
            print("Closing robot connection before exit...")
            try:
                result = device_close()
                print(result)
            except Exception as e:
                print(f"Error closing device: {e}")

            print("Exiting interactive session.")
            break
        print("\n================ USER PROMPT ================\n")
        print(user_text)

        # Add new user message and loop again
        messages.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

if __name__ == "__main__":
    main()

