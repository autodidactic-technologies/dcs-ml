# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"
#
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 75, 50, 30, 20, 10, 5])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     ego_altitude = random.randint(5000, 40000)  # ft
#     enemy_altitude = random.randint(5000, 40000)  # ft
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define prompt for GPT-4
#     prompt = (
#         f"Air combat scenario:\n"
#         f"- Distance to enemy: {distance} km\n"
#         f"- Ego heading: {ego_heading}°\n"
#         f"- Enemy heading: {enemy_heading}°\n"
#         f"- Ego altitude: {ego_altitude} ft\n"
#         f"- Enemy altitude: {enemy_altitude} ft\n"
#         f"- RWR {'active' if rwr_active else 'inactive'}\n"
#         f"- Enemy {'is engaging' if enemy_engaging else 'not engaging'}\n"
#         f"- F-16 Loadout: 2 AIM-120C, 2 AIM-9L\n"
#         f"- Su-30 Loadout: 2 R-77, 2 Archer IR\n\n"
#         f"Provide the best tactical action in JSON format with exactly one action per category:\n"
#         f'''
#         {{
#             "Maneuver": "<maneuver action>",
#             "Sensoring": "<sensoring action>",
#             "Firing": "<firing action>",
#             "Countermeasuring": "<countermeasure action>"
#         }}
#         '''
#     )
#
#     # Call OpenAI API
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an F-16 combat pilot providing structured tactical decisions. Keep responses precise and follow the JSON format strictly."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=150
#         )
#         action = response["choices"][0]["message"]["content"].strip()
#
#         # Ensure the response is valid JSON
#         try:
#             action_json = json.loads(action)
#         except json.JSONDecodeError:
#             action_json = {
#                 "Maneuver": "Turn defensive",
#                 "Sensoring": "Lock radar",
#                 "Firing": "Hold fire",
#                 "Countermeasuring": "Deploy chaff"
#             }
#
#     except Exception as e:
#         print(f"Error calling OpenAI: {e}")
#         action_json = {
#             "Maneuver": "Hold course",
#             "Sensoring": "Monitor RWR",
#             "Firing": "Do not engage",
#             "Countermeasuring": "No action"
#         }
#
#     return {"prompt": prompt, "completion": action_json}
#
# # Generate dataset
# dataset_size = 100  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_structured_dataset.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")
#



# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"  # Replace with your actual key
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 75, 50, 30, 20, 10, 5])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     ego_altitude = random.randint(5000, 40000)  # ft
#     enemy_altitude = random.randint(5000, 40000)  # ft
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define structured input
#     scenario = {
#         "Distance to enemy": f"{distance} km",
#         "Ego heading": f"{ego_heading}°",
#         "Enemy heading": f"{enemy_heading}°",
#         "Ego altitude": f"{ego_altitude} ft",
#         "Enemy altitude": f"{enemy_altitude} ft",
#         "RWR": "Active" if rwr_active else "Inactive",
#         "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
#         "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
#         "Su-30 Loadout": "2 R-77, 2 Archer IR"
#     }
#
#     # Define function to enforce JSON output
#     function_call = {
#         "name": "generate_tactical_decision",
#         "description": "Generate the best tactical decision for an F-16 combat scenario.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "Maneuver": {"type": "string", "description": "Best maneuvering action"},
#                 "Sensoring": {"type": "string", "description": "Best radar/sensor action"},
#                 "Firing": {"type": "string", "description": "Best missile/gun action"},
#                 "Countermeasuring": {"type": "string", "description": "Best countermeasure action"}
#             },
#             "required": ["Maneuver", "Sensoring", "Firing", "Countermeasuring"]
#         }
#     }
#
#     # Call OpenAI API with function calling
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an F-16 combat pilot. Respond with the best tactical decision in JSON format."},
#                 {"role": "user", "content": f"Scenario: {json.dumps(scenario, indent=2)}"}
#             ],
#             functions=[function_call],
#             function_call="auto",  # Forces JSON output
#             temperature=0.7,  # Ensures deterministic responses
#             max_tokens=150  # Prevents cutoff
#         )
#
#         # Extract the JSON output
#         action_json = response["choices"][0]["message"]["function_call"]["arguments"]
#         action_json = json.loads(action_json)  # Ensure it's parsed correctly
#
#     except Exception as e:
#         print(f"Error calling OpenAI: {e}")
#         action_json = {
#             "Maneuver": "Hold course",
#             "Sensoring": "Monitor RWR",
#             "Firing": "Do not engage",
#             "Countermeasuring": "No action"
#         }
#
#     return {"prompt": json.dumps(scenario, indent=2), "completion": action_json}
#
# # Generate dataset
# dataset_size = 100  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_function_calling_dataset.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")




# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 75, 50, 30, 20, 10, 5])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     ego_altitude = random.randint(5000, 40000)  # ft
#     enemy_altitude = random.randint(5000, 40000)  # ft
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define structured input
#     scenario = {
#         "Distance to enemy": f"{distance} km",
#         "Ego heading": f"{ego_heading}°",
#         "Enemy heading": f"{enemy_heading}°",
#         "Ego altitude": f"{ego_altitude} ft",
#         "Enemy altitude": f"{enemy_altitude} ft",
#         "RWR": "Active" if rwr_active else "Inactive",
#         "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
#         "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
#         "Su-30 Loadout": "2 R-77, 2 Archer IR"
#     }
#
#     # Define function to enforce JSON output with "No Action Required" option
#     function_call = {
#         "name": "generate_tactical_decision",
#         "description": "Generate the best tactical decision for an F-16 combat scenario following real-world air combat doctrine.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "Maneuver": {"type": "string", "description": "Best maneuvering action OR 'No action required'"},
#                 "Sensoring": {"type": "string", "description": "Best radar/sensor action OR 'No action required'"},
#                 "Firing": {"type": "string", "description": "Best weapon choice OR 'No action required'"},
#                 "Countermeasuring": {"type": "string", "description": "Best countermeasure action OR 'No action required'"}
#             },
#             "required": ["Maneuver", "Sensoring", "Firing", "Countermeasuring"]
#         }
#     }
#
#     # **Fixed system message enforcing tactical doctrine**
#     system_message = """You are an experienced F-16 combat pilot and instructor in modern air combat doctrine.
#     You MUST follow these rules when choosing actions:
#     - BVR (>50km): Use AIM-120C, prioritize high-energy shots, maintain separation.
#     - Mid-range (20-50km): AIM-120C still viable, but prepare for evasion.
#     - WVR (<20km): DO NOT use AIM-120C. Use AIM-9L for IR-guided engagement.
#     - Dogfight (<5km): AIM-9L or guns only. DO NOT suggest AIM-120C.
#     - If RWR is active and enemy is engaging, always consider countermeasures.
#     - If no action is required for a category, return 'No action required'.
#     - Only ONE action per category. Be precise.
#     """
#
#     # Call OpenAI API with function calling
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": f"Scenario: {json.dumps(scenario, indent=2)}"}
#             ],
#             functions=[function_call],
#             function_call="auto",  # Forces JSON output
#             temperature=0.0,  # Ensures deterministic responses
#             max_tokens=100  # Prevents cutoff
#         )
#
#         # Extract the JSON output
#         action_json = response["choices"][0]["message"]["function_call"]["arguments"]
#         action_json = json.loads(action_json)  # Ensure it's parsed correctly
#
#     except Exception as e:
#         print(f"Error calling OpenAI: {e}")
#         action_json = {
#             "Maneuver": "No action required",
#             "Sensoring": "No action required",
#             "Firing": "No action required",
#             "Countermeasuring": "No action required"
#         }
#
#     return {"prompt": json.dumps(scenario, indent=2), "completion": action_json}
#
# # Generate dataset
# dataset_size = 1000  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_tactical_no_action_dataset.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")










# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"  # Replace with your actual key
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 75, 50, 30, 20, 10, 5])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     ego_altitude = random.randint(5000, 40000)  # ft
#     enemy_altitude = random.randint(5000, 40000)  # ft
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define structured input
#     scenario = {
#         "Distance to enemy": f"{distance} km",
#         "Ego heading": f"{ego_heading}°",
#         "Enemy heading": f"{enemy_heading}°",
#         "Ego altitude": f"{ego_altitude} ft",
#         "Enemy altitude": f"{enemy_altitude} ft",
#         "RWR": "Active" if rwr_active else "Inactive",
#         "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
#         "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
#         "Su-30 Loadout": "2 R-77, 2 Archer IR"
#     }
#
#     # Free-form question to GPT
#     tactical_prompt = f"""
#     You are an experienced F-16 combat pilot engaging a Su-30 in aerial combat.
#
#     Scenario:
#     - Distance to enemy: {distance} km
#     - Ego heading: {ego_heading}°
#     - Enemy heading: {enemy_heading}°
#     - Ego altitude: {ego_altitude} ft
#     - Enemy altitude: {enemy_altitude} ft
#     - RWR: {'Active' if rwr_active else 'Inactive'}
#     - Enemy: {'Engaging' if enemy_engaging else 'Not engaging'}
#     - F-16 Loadout: 2 AIM-120C, 2 AIM-9L
#     - Su-30 Loadout: 2 R-77, 2 Archer IR
#
#     What is the best tactical action? Respond naturally as if advising a pilot.
#     """
#
#     # Call OpenAI API for tactical response
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an experienced F-16 combat pilot providing tactical decisions."},
#                 {"role": "user", "content": tactical_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=200
#         )
#         raw_response = response["choices"][0]["message"]["content"].strip()
#
#     except Exception as e:
#         print(f"Error calling OpenAI for response: {e}")
#         raw_response = "Maintain position and monitor enemy movements."
#
#     # Step 2: Ask GPT to interpret its own response and map to categories
#     interpretation_prompt = f"""
#     You are an expert in military air combat tactics. Below is a tactical response given by an F-16 pilot.
#     Your task is to interpret it and categorize the actions into four categories:
#     - Maneuver
#     - Sensoring
#     - Firing
#     - Countermeasuring
#
#     If a category does not apply, return "No action required."
#
#     Tactical Response:
#     "{raw_response}"
#
#     Provide the structured output in this JSON format:
#     {{
#         "Maneuver": "<Extracted maneuver action or 'No action required'>",
#         "Sensoring": "<Extracted sensor action or 'No action required'>",
#         "Firing": "<Extracted firing action or 'No action required'>",
#         "Countermeasuring": "<Extracted countermeasure action or 'No action required'>"
#     }}
#     """
#
#     try:
#         interpretation_response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a military air combat AI that classifies tactical responses."},
#                 {"role": "user", "content": interpretation_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=150
#         )
#         classified_actions = json.loads(interpretation_response["choices"][0]["message"]["content"].strip())
#
#     except Exception as e:
#         print(f"Error calling OpenAI for interpretation: {e}")
#         classified_actions = {
#             "Maneuver": "No action required",
#             "Sensoring": "No action required",
#             "Firing": "No action required",
#             "Countermeasuring": "No action required"
#         }
#
#     return {"prompt": tactical_prompt, "raw_response": raw_response, "completion": classified_actions}
#
#
# # Generate dataset
# dataset_size = 100  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_self_interpretation_dataset.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")



#
# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"  # Replace with your actual key
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 75, 50, 30, 20, 10, 5])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     ego_altitude = random.randint(5000, 40000)  # ft
#     enemy_altitude = random.randint(5000, 40000)  # ft
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define structured input
#     scenario = {
#         "Distance to enemy": f"{distance} km",
#         "Ego heading": f"{ego_heading}°",
#         "Enemy heading": f"{enemy_heading}°",
#         "Ego altitude": f"{ego_altitude} ft",
#         "Enemy altitude": f"{enemy_altitude} ft",
#         "RWR": "Active" if rwr_active else "Inactive",
#         "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
#         "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
#         "Su-30 Loadout": "2 R-77, 2 Archer IR"
#     }
#
#     # Free-form question to GPT
#     tactical_prompt = f"""
#     You are an experienced F-16 combat pilot engaging a Su-30 in aerial combat.
#
#     Scenario:
#     - Distance to enemy: {distance} km
#     - Ego heading: {ego_heading}°
#     - Enemy heading: {enemy_heading}°
#     - Ego altitude: {ego_altitude} ft
#     - Enemy altitude: {enemy_altitude} ft
#     - RWR: {'Active' if rwr_active else 'Inactive'}
#     - Enemy: {'Engaging' if enemy_engaging else 'Not engaging'}
#     - F-16 Loadout: 2 AIM-120C, 2 AIM-9L
#     - Su-30 Loadout: 2 R-77, 2 Archer IR
#
#     What is the best tactical action? Respond naturally as if advising a pilot.
#     """
#
#     # Call OpenAI API for tactical response
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an experienced F-16 combat pilot providing tactical decisions."},
#                 {"role": "user", "content": tactical_prompt}
#             ],
#             temperature=0.8,
#             max_tokens=150  # Reduced to prevent truncation
#         )
#         raw_response = response["choices"][0]["message"]["content"].strip()
#
#     except Exception as e:
#         print(f"Error calling OpenAI for response: {e}")
#         raw_response = "Maintain position and monitor enemy movements."
#
#     # Step 2: Ask GPT to interpret its own response and map to categories using function calling
#     interpretation_prompt = f"""
#     You are a military AI that classifies air combat tactical decisions.
#
#     Below is a tactical response given by an F-16 pilot.
#     Your task is to interpret it and categorize the actions into four categories:
#     - Maneuver
#     - Sensoring
#     - Firing
#     - Countermeasuring
#
#     If a category does not apply, return "No action required."
#
#     Tactical Response:
#     "{raw_response}"
#
#     Provide the structured output in valid JSON:
#     {{
#         "Maneuver": "<Extracted maneuver action or 'No action required'>",
#         "Sensoring": "<Extracted sensor action or 'No action required'>",
#         "Firing": "<Extracted firing action or 'No action required'>",
#         "Countermeasuring": "<Extracted countermeasure action or 'No action required'>"
#     }}
#     """
#
#     try:
#         interpretation_response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an AI that classifies tactical responses into predefined categories."},
#                 {"role": "user", "content": interpretation_prompt}
#             ],
#             temperature=0.8,
#             max_tokens=100
#         )
#
#         # Extract JSON output
#         classified_actions = json.loads(interpretation_response["choices"][0]["message"]["content"].strip())
#
#     except Exception as e:
#         print(f"Error calling OpenAI for interpretation: {e}")
#         classified_actions = {
#             "Maneuver": "No action required",
#             "Sensoring": "No action required",
#             "Firing": "No action required",
#             "Countermeasuring": "No action required"
#         }
#
#     return {
#         "prompt": tactical_prompt,
#         "raw_response": raw_response,
#         "completion": classified_actions
#     }
#
#
# # Generate dataset
# dataset_size = 100  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_fixed_interpretation.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")








# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"  # Replace with your actual key
#
# # Predefined action space
# MANEUVER_ACTIONS = [
#     "Maintain course", "Crank left", "Crank right", "Dive for speed",
#     "Climb for energy", "Notch left", "Notch right", "Extend away"
# ]
#
# SENSORING_ACTIONS = [
#     "Lock target STT", "Lock target TWS", "Scan with radar",
#     "Monitor RWR", "Passive detection"
# ]
#
# FIRING_ACTIONS = [
#     "Fire AIM-120C", "Fire AIM-9L", "Gun attack", "Hold fire"
# ]
#
# COUNTERMEASURING_ACTIONS = [
#     "Deploy chaff", "Deploy flares", "ECM jamming", "Break lock"
# ]
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 90, 75, 60, 50, 30, 20, 10, 5, 2])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     ego_altitude = random.randint(5000, 40000)  # ft
#     enemy_altitude = random.randint(5000, 40000)  # ft
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define structured input
#     scenario = {
#         "Distance to enemy": f"{distance} km",
#         "Ego heading": f"{ego_heading}°",
#         "Enemy heading": f"{enemy_heading}°",
#         "Ego altitude": f"{ego_altitude} ft",
#         "Enemy altitude": f"{enemy_altitude} ft",
#         "RWR": "Active" if rwr_active else "Inactive",
#         "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
#         "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
#         "Su-30 Loadout": "2 R-77, 2 Archer IR"
#     }
#
#     # Free-form tactical response request
#     tactical_prompt = f"""
#     You are an experienced F-16 combat pilot engaging a Su-30 in aerial combat.
#
#     Scenario:
#     {json.dumps(scenario, indent=2)}
#
#     What is the best tactical action? Respond naturally as if advising a pilot.
#     """
#
#     # Call OpenAI API for tactical response
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are an experienced F-16 combat pilot providing tactical decisions."},
#                 {"role": "user", "content": tactical_prompt}
#             ],
#             temperature=0.8,
#             max_tokens=250
#         )
#         raw_response = response["choices"][0]["message"]["content"].strip()
#
#     except Exception as e:
#         print(f"Error calling OpenAI for response: {e}")
#         raw_response = "Maintain position and monitor enemy movements."
#
#     # Step 2: Map response to predefined actions
#     classified_actions = map_response_to_actions(raw_response)
#
#     return {
#         "prompt": tactical_prompt,
#         "raw_response": raw_response,
#         "completion": classified_actions
#     }
#
#
# # Function to map GPT's response to predefined action space
# def map_response_to_actions(response_text):
#     """Interprets GPT's free-form response and maps it to predefined action categories"""
#     maneuver = next((action for action in MANEUVER_ACTIONS if action.lower() in response_text.lower()), "Maintain course")
#     sensoring = next((action for action in SENSORING_ACTIONS if action.lower() in response_text.lower()), "Monitor RWR")
#     firing = next((action for action in FIRING_ACTIONS if action.lower() in response_text.lower()), "Hold fire")
#     countermeasuring = next((action for action in COUNTERMEASURING_ACTIONS if action.lower() in response_text.lower()), "No action required")
#
#     return {
#         "Maneuver": maneuver,
#         "Sensoring": sensoring,
#         "Firing": firing,
#         "Countermeasuring": countermeasuring
#     }
#
#
# # Generate dataset
# dataset_size = 100  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_action_space.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")














#
# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"  # Replace with your actual key
#
# # Define the predefined tactical action space
# ACTION_SPACE = {
#     "Maneuver": [
#         "Maintain course", "Crank left", "Crank right", "Dive for speed",
#         "Climb for energy", "Notch left", "Notch right", "Extend away"
#     ],
#     "Sensoring": [
#         "Lock target STT", "Lock target TWS", "Scan with radar",
#         "Monitor RWR", "Passive detection"
#     ],
#     "Firing": [
#         "Fire AIM-120C", "Fire AIM-9L", "Gun attack", "Hold fire"
#     ],
#     "Countermeasuring": [
#         "Deploy chaff", "Deploy flares", "ECM jamming", "Break lock", "No action required"
#     ]
# }
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 90, 75, 60, 50, 30, 20, 10, 5, 2])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define structured input (REMOVED ALTITUDE FACTOR)
#     scenario = {
#         "Distance to enemy": f"{distance} km",
#         "Ego heading": f"{ego_heading}°",
#         "Enemy heading": f"{enemy_heading}°",
#         "RWR": "Active" if rwr_active else "Inactive",
#         "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
#         "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
#         "Su-30 Loadout": "2 R-77, 2 Archer IR"
#     }
#
#     # Free-form tactical response request
#     tactical_prompt = f"""
#     You are an experienced F-16 combat pilot engaging a Su-30 in aerial combat.
#
#     Scenario:
#     {json.dumps(scenario, indent=2)}
#
#     What is the best tactical action? Respond naturally as if advising a pilot.
#     """
#
#     # Call OpenAI API for tactical response
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an experienced F-16 combat pilot providing tactical decisions."},
#                 {"role": "user", "content": tactical_prompt}
#             ],
#             temperature=0.8,
#             max_tokens=250
#         )
#         raw_response = response["choices"][0]["message"]["content"].strip()
#
#     except Exception as e:
#         print(f"Error calling OpenAI for response: {e}")
#         raw_response = "Maintain position and monitor enemy movements."
#
#     # Step 2: Ask GPT to interpret its own response and map to predefined actions
#     classified_actions = interpret_response_with_gpt(raw_response)
#
#     return {
#         "prompt": tactical_prompt,
#         "raw_response": raw_response,
#         "completion": classified_actions
#     }
#
#
# # Function to interpret GPT's response and map it to predefined action space
# def interpret_response_with_gpt(response_text):
#     """Uses GPT to classify its own tactical response into predefined categories"""
#
#     interpretation_prompt = f"""
#     You are an AI that classifies air combat tactical decisions.
#
#     Below is a tactical response given by an F-16 pilot.
#     Your task is to interpret it and categorize the actions into four categories:
#     - Maneuver
#     - Sensoring
#     - Firing
#     - Countermeasuring
#
#     If a category does not apply, return "No action required."
#
#     Tactical Response:
#     "{response_text}"
#
#     Choose the most appropriate predefined action from the following action space:
#
#     Maneuver Options: {ACTION_SPACE["Maneuver"]}
#     Sensoring Options: {ACTION_SPACE["Sensoring"]}
#     Firing Options: {ACTION_SPACE["Firing"]}
#     Countermeasuring Options: {ACTION_SPACE["Countermeasuring"]}
#
#     Provide the structured output in valid JSON:
#     {{
#         "Maneuver": "<Selected maneuver action>",
#         "Sensoring": "<Selected sensor action>",
#         "Firing": "<Selected firing action>",
#         "Countermeasuring": "<Selected countermeasure action>"
#     }}
#     """
#
#     try:
#         interpretation_response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a military air combat AI that classifies tactical responses into predefined categories."},
#                 {"role": "user", "content": interpretation_prompt}
#             ],
#             temperature=0.2,
#             max_tokens=100
#         )
#
#         # Extract JSON output
#         classified_actions = json.loads(interpretation_response["choices"][0]["message"]["content"].strip())
#
#     except Exception as e:
#         print(f"Error calling OpenAI for interpretation: {e}")
#         classified_actions = {
#             "Maneuver": "No action required",
#             "Sensoring": "No action required",
#             "Firing": "No action required",
#             "Countermeasuring": "No action required"
#         }
#
#     return classified_actions
#
#
# # Generate dataset
# dataset_size = 100  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_self_interpretation.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")














# import openai
# import json
# import random
# import time
#
# # Set your OpenAI API key
# openai.api_key = "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"  # Replace with your actual key
#
# # Define the predefined tactical action space
# ACTION_SPACE = {
#     "Maneuver": [
#         "Maintain course", "Crank left", "Crank right", "Dive for speed",
#         "Climb for energy", "Notch left", "Notch right", "Extend away"
#     ],
#     "Sensoring": [
#         "Lock target STT", "Lock target TWS", "Scan with radar",
#         "Monitor RWR", "Passive detection"
#     ],
#     "Firing": [
#         "Fire AIM-120C", "Fire AIM-9L", "Gun attack", "Hold fire"
#     ],
#     "Countermeasuring": [
#         "Deploy chaff", "Deploy flares", "ECM jamming", "Break lock", "No action required"
#     ]
# }
#
# # Function to generate a combat scenario
# def generate_scenario():
#     """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""
#
#     # Randomized scenario variables
#     distance = random.choice([100, 90, 75, 60, 50, 30, 20, 10, 5, 2])  # km
#     ego_heading = random.randint(0, 360)
#     enemy_heading = random.randint(0, 360)
#     rwr_active = random.choice([True, False])
#     enemy_engaging = random.choice([True, False])
#
#     # Define structured input (REMOVED ALTITUDE FACTOR)
#     scenario = {
#         "Distance to enemy": f"{distance} km",
#         "Ego heading": f"{ego_heading}°",
#         "Enemy heading": f"{enemy_heading}°",
#         "RWR": "Active" if rwr_active else "Inactive",
#         "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
#         "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
#         "Su-30 Loadout": "2 R-77, 2 Archer IR"
#     }
#
#     # Call OpenAI API for structured tactical response using Function Calling
#     classified_actions = interpret_response_with_gpt(scenario)
#
#     return {
#         "prompt": json.dumps(scenario, indent=2),
#         "completion": classified_actions
#     }
#
#
# # Function to interpret GPT's response and map it to predefined action space using Function Calling
# def interpret_response_with_gpt(scenario):
#     """Uses GPT to classify tactical decisions with enforced JSON output using Function Calling"""
#
#     function_call = {
#         "name": "classify_tactical_response",
#         "description": "Categorize the best tactical response for an F-16 pilot in a combat scenario.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "Maneuver": {"type": "string", "enum": ACTION_SPACE["Maneuver"]},
#                 "Sensoring": {"type": "string", "enum": ACTION_SPACE["Sensoring"]},
#                 "Firing": {"type": "string", "enum": ACTION_SPACE["Firing"]},
#                 "Countermeasuring": {"type": "string", "enum": ACTION_SPACE["Countermeasuring"]}
#             },
#             "required": ["Maneuver", "Sensoring", "Firing", "Countermeasuring"]
#         }
#     }
#
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an AI that classifies air combat tactical decisions."},
#                 {"role": "user", "content": f"Scenario: {json.dumps(scenario, indent=2)}"}
#             ],
#             functions=[function_call],  # Enforce function calling
#             function_call="auto",  # Ensure GPT returns structured JSON
#             temperature=0.8,  # Enforce consistency
#             max_tokens=100
#         )
#
#         # Extract JSON response
#         action_json = response["choices"][0]["message"]["function_call"]["arguments"]
#         action_json = json.loads(action_json)  # Ensure it's parsed correctly
#
#     except Exception as e:
#         print(f"Error calling OpenAI for interpretation: {e}")
#         action_json = {
#             "Maneuver": "No action required",
#             "Sensoring": "No action required",
#             "Firing": "No action required",
#             "Countermeasuring": "No action required"
#         }
#
#     return action_json
#
#
# # Generate dataset
# dataset_size = 100  # Number of samples
# dataset = []
#
# for _ in range(dataset_size):
#     print(_)
#     scenario = generate_scenario()
#     print(scenario)
#     dataset.append(scenario)
#     time.sleep(1)  # Prevent API rate limit issues
#
# # Save dataset to JSONL format
# output_file = "dcs_gpt4_function_calling.jsonl"
# with open(output_file, "w") as f:
#     for entry in dataset:
#         f.write(json.dumps(entry) + "\n")
#
# print(f"Dataset of {dataset_size} samples saved to {output_file}")












import openai
import json
import random
import time

# Set your OpenAI API key
openai.api_key = "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"  # Replace with your actual key

# Define the predefined tactical action space
ACTION_SPACE = {
    "Maneuver": [
        "Maintain course", "Crank left", "Crank right", "Dive for speed",
        "Climb for energy", "Notch left", "Notch right", "Extend away"
    ],
    "Sensoring": [
        "Lock target STT", "Lock target TWS", "Scan with radar",
        "Monitor RWR", "Passive detection"
    ],
    "Firing": [
        "Fire AIM-120C", "Fire AIM-9L", "Gun attack", "Hold fire"
    ],
    "Countermeasuring": [
        "Deploy chaff", "Deploy flares", "ECM jamming", "Break lock", "No action required"
    ]
}

# Function to generate a combat scenario
def generate_scenario():
    """Generates a structured air combat engagement scenario with F-16 vs. Su-30"""

    # Randomized scenario variables
    distance = random.choice([100, 75, 50, 30, 20, 10, 5])  # km
    ego_heading = random.randint(0, 360)
    enemy_heading = random.randint(0, 360)
    rwr_active = random.choice([True, False])
    enemy_engaging = random.choice([True, False])

    # Define structured input (REMOVED ALTITUDE FACTOR)
    scenario = {
        "Distance to enemy": f"{distance} km",
        "Ego heading": f"{ego_heading}°",
        "Enemy heading": f"{enemy_heading}°",
        "RWR": "Active" if rwr_active else "Inactive",
        "Enemy status": "Engaging" if enemy_engaging else "Not engaging",
        "F-16 Loadout": "2 AIM-120C, 2 AIM-9L",
        "Su-30 Loadout": "2 R-77, 2 Archer IR"
    }

    # Free-form tactical response request
    tactical_prompt = f"""
    You are an experienced F-16 combat pilot engaging a Su-30 in aerial combat.

    Scenario:
    {json.dumps(scenario, indent=2)}

    What is the best tactical action? Respond naturally as if advising a pilot.
    """

    # Step 1: Call OpenAI API for free-form tactical response
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced F-16 combat pilot providing tactical decisions."},
                {"role": "user", "content": tactical_prompt}
            ],
            temperature=0.8,
            max_tokens=150
        )
        raw_response = response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Error calling OpenAI for response: {e}")
        raw_response = "Maintain position and monitor enemy movements."

    # Step 2: Ask GPT to interpret its own response and map to predefined actions
    classified_actions = interpret_response_with_gpt(raw_response)

    return {
        "prompt": tactical_prompt,
        "raw_response": raw_response,
        "completion": classified_actions
    }


# Function to interpret GPT's response and map it to predefined action space
def interpret_response_with_gpt(response_text):
    """Uses GPT to classify its own response into predefined categories"""

    interpretation_prompt = f"""
    You are an AI that classifies air combat tactical decisions.

    Below is a tactical response given by an F-16 pilot. 
    Your task is to interpret it and categorize the actions into four categories:
    - Maneuver
    - Sensoring
    - Firing
    - Countermeasuring

    Choose the **closest** predefined action for each category from this list:

    Maneuver Options: {ACTION_SPACE["Maneuver"]}
    Sensoring Options: {ACTION_SPACE["Sensoring"]}
    Firing Options: {ACTION_SPACE["Firing"]}
    Countermeasuring Options: {ACTION_SPACE["Countermeasuring"]}

    If no action applies, return "No action required."

    Tactical Response:
    "{response_text}"

    Provide the structured output in **valid JSON**:
    {{
        "Maneuver": "<Selected maneuver action>",
        "Sensoring": "<Selected sensor action>",
        "Firing": "<Selected firing action>",
        "Countermeasuring": "<Selected countermeasure action>"
    }}
    """

    try:
        interpretation_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a military air combat AI that classifies tactical responses into predefined categories."},
                {"role": "user", "content": interpretation_prompt}
            ],
            temperature=0.8,
            max_tokens=100
        )

        # Extract JSON response
        classified_actions = json.loads(interpretation_response["choices"][0]["message"]["content"].strip())

    except Exception as e:
        print(f"Error calling OpenAI for interpretation: {e}")
        classified_actions = {
            "Maneuver": "No action required",
            "Sensoring": "No action required",
            "Firing": "No action required",
            "Countermeasuring": "No action required"
        }

    return classified_actions


# Generate dataset
dataset_size = 100  # Number of samples
dataset = []

for _ in range(dataset_size):
    print(_)
    scenario = generate_scenario()
    print(scenario)
    dataset.append(scenario)
    time.sleep(1)  # Prevent API rate limit issues

# Save dataset to JSONL format
output_file = "dcs_gpt4_freedom_with_mapping.jsonl"
with open(output_file, "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

print(f"Dataset of {dataset_size} samples saved to {output_file}")
