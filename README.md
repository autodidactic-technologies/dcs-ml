- The mediator.py acts like a supervisor that checks the agent’s actions using a language model and can change them if needed. feature_translator.py converts the environment state into text. minigrid_agent_prompt.py builds the prompts sent to the LLM.
- The minigrid_assistant_mediator.py handles the replies from the language model and helps decide if the action should change.
- To train the mediator run medaitor_main_training.py (Select your configraiton it can also evaluate the agent).

 