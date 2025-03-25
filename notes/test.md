- **<a href="https://github.com/openai/openai-agents-python" target="_blank">OpenAI Agents SDK</a>**
    - **Overview** <!-- markmap: fold -->
        - lets you:
            - build **agentic AI apps**
            - use a lightweight, easy-to-use package with very few abstractions
            - production-ready upgrade of <a href="https://github.com/openai/swarm/tree/main" target="_blank">Swarm</a>
        - introduces very small set of primitives **$\Rightarrow$ Python + SDK primitives**:
            - Existing Python primitives
            - **3 new SDK primitives**:
                - **Agents**: LLMs equipped with instructions and tools
                - **Handoffs**: Allow agents to delegate to other agents for specific tasks
                - **Guardrails**: Enable the inputs to agents to be validated
            - Powerful enough to:
                - express complex **relationships b/w tools & agents**
                - let you **build real-world applications** without a steep learning curve
        - SDK has 2 driving **design principles**:
            - **Enough features** $\rightarrow$ **worth using**, **Few enough** primitives $\rightarrow$ **quick to learn**.
            - Works great **out of the box**, but you can **customize** exactly what happens.
        - Main features of the SDK:
            - **Agent loop**: Built-in agent loop that handles:
                - calling tools
                - sending results to the LLM
                - looping until the LLM is done
            - **Python-first**: Use built-in language features to **orchestrate and chain agents**, rather than needing to learn new abstractions.
            - **Handoffs**: A powerful feature to **coordinate and delegate** between multiple agents.
            - **Guardrails**: Run input validations and checks in parallel to your agents, breaking early if the checks fail.
            - **Function tools**: Turn any Python function into a tool, with automatic schema generation and Pydantic-powered validation.
            - **Tracing**: Built-in tracing that lets you:
                - **visualize** your agentic workflow
                - **debug** and **monitor** your workflows
                - use the OpenAI suite of:
                    - **evaluation** tools
                    - **fine-tuning** tools
                    - **distillation** tools
                - integrate with external processors like Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI
    - **Example**<br> <!-- markmap: fold -->
        **Multi-agent workflow**
        **with handoffs & guardrails**:
        - **Setup**:
            ```bash
            mkdir my_project && cd my_project
            uv venv && source .venv/bin/activate
            echo "openai-agents" > requirements.txt && uv pip install -r requirements.txt
            export OPENAI_API_KEY=sk-...
            ```
        - **Imports**:
            ```python
            from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
            from pydantic import BaseModel
            import asyncio
            ```
        - **Create multiple agents and define handoffs**:
            ```python
            # Define guardrail output structure
            class HomeworkOutput(BaseModel):
                is_homework: bool
                reasoning: str

            math_tutor_agent = Agent(
                name="Math Tutor",
                handoff_description="Specialist agent for math questions",
                instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
            )

            history_tutor_agent = Agent(
                name="History Tutor",
                handoff_description="Specialist agent for historical questions",
                instructions="You provide assistance with historical queries. Explain important events and context clearly.",
            )
            ```
        - **Add guardrail**:
            ```python
            # Define guardrail agent
            guardrail_agent = Agent(
                name="Guardrail check",
                instructions="Check if the user is asking about homework.",
                output_type=HomeworkOutput,
            )
            
            async def homework_guardrail(ctx, agent, input_data):
                result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
                final_output = result.final_output_as(HomeworkOutput)
                return GuardrailFunctionOutput(
                    output_info=final_output,
                    tripwire_triggered=not final_output.is_homework,
                )

            triage_agent = Agent(
                name="Triage Agent",
                instructions="You determine which agent to use based on the user's homework question",
                handoffs=[history_tutor_agent, math_tutor_agent],
                input_guardrails=[
                    InputGuardrail(guardrail_function=homework_guardrail),
                ],
            )
            ```
        - **Run the agent orchestration**:
            ```python
            async def main():
                # Run with a history question
                result = await Runner.run(triage_agent, "Who was the first president of the United States?")
                print(result.final_output)

                # Run with a non-homework question
                result = await Runner.run(triage_agent, "What is life")
                print(result.final_output)

            if __name__ == "__main__":
                asyncio.run(main())
            ```
    - **Examples** <!-- markmap: fold -->
        - **Common agent design patterns**:
            - **Deterministic workflows**: Structuring agents for predictable, linear execution paths
            - **Agents as tools**: Using agents as tools for other agents
            - **Parallel agent execution**: Running multiple agents concurrently
        
        - **SDK foundational capabilities**:
            - **Dynamic system prompts**: Customizing agent instructions at runtime
            - **Streaming outputs**: Processing agent responses in real-time
            - **Lifecycle events**: Hooking into agent execution lifecycle
        
        - **Tool Examples**:
            - Integration with OAI hosted tools like **web search** and **file search**
        
        - **Model Providers**:
            - Using **non-OpenAI models** with the SDK
        
        - **Handoffs**:
            - Practical examples of **agent-to-agent delegation**
        
        - **Real-world Applications**:
            - **Customer Service**: Example customer service system for an airline
            - **Research Bot**: Simple deep research clone
    - **Agents**:
        - **Agents** are the core building blocks in your apps. 
            They are **LLMs** configured with **instructions** and **tools**.
        
        - **Basic configuration**:
            - **Name**: Identifier for the agent
            - **Instructions**: System prompt/developer message
            - **Tools**: Functions the agent can use 
            - **Model**: Which LLM to use
            - **ModelSettings** (optional): Controls LLM parameters <!-- 🚨 **TODO: CONFIRM THESE LATER** -->
                - **temperature**: Controls randomness (higher = more random)
                - **top_p**: Controls diversity
                - **frequency_penalty**: Penalizes repeated tokens
                - **presence_penalty**: Penalizes tokens based on presence
                - **tool_choice**: Can be "auto", "required", "none", or a specific tool name
                - **parallel_tool_calls**: Allow parallel tool execution
                - **truncation**: "auto" or "disabled"
                - **max_tokens**: Max output tokens to generate
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    from agents import Agent, ModelSettings, function_tool

                    @function_tool 
                    def get_weather(city: str) -> str: 
                        return f"The weather in {city} is sunny"

                    model_settings = ModelSettings(
                        temperature=0.7,
                        top_p=0.95,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        tool_choice="auto",
                        parallel_tool_calls=True,
                        truncation="auto",
                        max_tokens=None
                    )

                    agent = Agent(
                        name="Weather agent",
                        instructions="Always respond in haiku form",
                        tools=[get_weather],
                        model="o3-mini",
                        model_settings=model_settings
                    )
                    ```
        
        - **Context**:
            - Agents are generic on their **`context` type**.
            - **Context** is a **dependency-injection tool**:
                - It's an object you **create** and **pass** to `Runner.run()`
                - It is **passed to every**:
                    - **agent**
                    - **tool**
                    - **handoff**
                    - etc.
                - It serves as a **grab bag of dependencies and state** for the agent run
            - You can provide **any Python object** as the context.
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    @dataclass
                    class UserContext:
                      uid: str
                      is_pro_user: bool

                      async def fetch_purchases() -> list[Purchase]:
                         return ...

                    agent = Agent[UserContext](
                        ...,
                    )
                    ```
        
        - **Output types**:
            - By default, agents produce **plain text** (i.e. `str`) outputs.
            - Use the **`output_type` parameter** for **structured outputs**:
                - [**Pydantic**](https://docs.pydantic.dev/) objects
                - **dataclasses**
                - **lists**
                - **TypedDict**
                - Any type compatible with Pydantic [**TypeAdapter**](https://docs.pydantic.dev/latest/api/type_adapter/)
            - When you pass `output_type`, the model uses [**structured outputs**](https://platform.openai.com/docs/guides/structured-outputs) instead of plain text.
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    from pydantic import BaseModel
                    from agents import Agent

                    class CalendarEvent(BaseModel):
                        name: str
                        date: str
                        participants: list[str]

                    agent = Agent(
                        name="Calendar extractor",
                        instructions="Extract calendar events from text",
                        output_type=CalendarEvent,
                    )
                    ```
        
        - **Handoffs**:
            - **Handoffs** are **sub-agents** that the main agent can **delegate to**.
            - You provide a **list of handoffs**, and the agent chooses when to delegate.
            - This pattern enables **modular, specialized agents** that excel at specific tasks.
            - Read more in the [handoffs documentation](http://openai.github.io/openai-agents-python/handoffs/).
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    from agents import Agent

                    booking_agent = Agent(...)
                    refund_agent = Agent(...)

                    triage_agent = Agent(
                        name="Triage agent",
                        instructions=(
                            "Help the user with their questions."
                            "If they ask about booking, handoff to the booking agent."
                            "If they ask about refunds, handoff to the refund agent."
                        ),
                        handoffs=[booking_agent, refund_agent],
                    )
                    ```
        
        - **Dynamic instructions**:
            - Instead of **static instructions**, you can provide a **function**.
            - The function receives the **agent** and **context** and returns the prompt.
            - Both **regular** and **`async` functions** are supported.
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    def dynamic_instructions(
                        context: RunContextWrapper[UserContext], 
                        agent: Agent[UserContext]
                    ) -> str:
                        # Generate instructions based on context
                        return f"The user's name is {context.context.uid}. Help them with their questions."
                    
                    agent = Agent[UserContext](
                        name="Dynamic Agent",
                        instructions=dynamic_instructions,
                    )
                    
                    # Async function also works
                    async def async_instructions(context, agent) -> str:
                        user_data = await fetch_user_data(context.context.uid)
                        return f"User preferences: {user_data['preferences']}. Respond accordingly."
                    ```
        
        - **Lifecycle events (hooks)**:
            - Observe the **lifecycle** of an agent for **logging** or **pre-fetching** data.
            - Use the **`hooks` property** with a custom **AgentHooks** subclass.
            - Override the methods you're interested in.
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    from agents import Agent, AgentHooks, AgentResult, RunContextWrapper
                    
                    # Custom hooks implementation
                    class CustomHooks(AgentHooks):
                        def on_agent_start(self, context, agent) -> None:
                            print(f"Agent {agent.name} started")
                        
                        async def on_agent_finish(
                            self, 
                            context: RunContextWrapper, 
                            agent: Agent, 
                            result: AgentResult
                        ) -> None:
                            print(f"Agent {agent.name} finished with: {result.final_output[:50]}...")
                    
                    agent = Agent(
                        name="Hooked Agent",
                        instructions="Be helpful and concise",
                        hooks=CustomHooks(),
                    )
                    ```
        
        - **Guardrails**:
            - Run **validations** on user input **in parallel** to the agent.
            - Examples: screen for **relevance** or check if asking for **homework solutions**.
            - Read more in the [guardrails documentation](http://openai.github.io/openai-agents-python/guardrails/).
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
                    from pydantic import BaseModel
                    
                    # Define guardrail output structure
                    class HomeworkOutput(BaseModel):
                        is_homework: bool
                        reasoning: str
                    
                    # Define guardrail agent
                    guardrail_agent = Agent(
                        name="Guardrail checker",
                        instructions="Check if the user is asking about homework.",
                        output_type=HomeworkOutput,
                    )
                    
                    # Guardrail function to check input
                    async def homework_guardrail(ctx, agent, input_data):
                        result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
                        final_output = result.final_output_as(HomeworkOutput)
                        
                        return GuardrailFunctionOutput(
                            output_info=final_output,
                            tripwire_triggered=final_output.is_homework,
                        )
                    
                    # Create agent with guardrail
                    agent = Agent(
                        name="Tutor Agent",
                        instructions="Help with questions, but don't solve homework",
                        input_guardrails=[
                            InputGuardrail(guardrail_function=homework_guardrail),
                        ],
                    )
                    ```
        
        - **Cloning/copying agents**:
            - The **`clone()`** method lets you **duplicate** an Agent.
            - You can **optionally change** any properties in the clone.
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    from agents import Agent
                    
                    # Create original agent
                    pirate_agent = Agent(
                        name="Pirate",
                        instructions="Write like a pirate",
                        model="o3-mini",
                    )
                    
                    # Clone agent with modifications
                    robot_agent = pirate_agent.clone(
                        name="Robot",
                        instructions="Write like a robot",
                    )
                    ```
        
        - **Forcing tool use**:
            - Control tool usage with **`ModelSettings.tool_choice`**.
            - Options:
                1. **`auto`**: LLM decides whether to use a tool
                2. **`required`**: LLM must use a tool (can decide which one)
                3. **`none`**: LLM must not use a tool
                4. **Specific tool name** (e.g., `"my_tool"`): LLM must use that tool
            - Consider setting **`Agent.tool_use_behavior`** to stop the Agent after tool output to prevent infinite loops.
            - **Example**: <!-- markmap: fold -->
                - **Code**:
                    ```python
                    from agents import Agent, ModelSettings, function_tool
                    
                    @function_tool
                    def search_database(query: str) -> str:
                        return f"Results for {query}: ..."
                    
                    # Create agent that must use tools
                    agent = Agent(
                        name="Research Agent",
                        instructions="Research topics using the search tool",
                        tools=[search_database],
                        model_settings=ModelSettings(
                            tool_choice="required",
                        ),
                        tool_use_behavior="stop_on_tool",
                    )
                    
                    # Create agent that must use a specific tool
                    agent_specific = Agent(
                        name="Search Agent",
                        instructions="Search for information",
                        tools=[search_database],
                        model_settings=ModelSettings(
                            tool_choice="search_database",
                        ),
                    )
                    ```
    - **Organize me later**: <!-- markmap: fold -->
        - **Examples**:
            - See the [examples in GitHub repo](https://github.com/openai/openai-agents-python/tree/main/examples) for complete implementations
        - **Quick Start**:
            - **Tracing**: View agent runs on [OpenAI Dashboard](https://platform.openai.com/traces)
            - **Next steps**: Learn about:
                - [Agent configuration](http://openai.github.io/openai-agents-python/agents/)
                - [Running agents](http://openai.github.io/openai-agents-python/running_agents/)
                - [Tools](http://openai.github.io/openai-agents-python/tools/)
                - [Guardrails](http://openai.github.io/openai-agents-python/guardrails/)
                - [Models](http://openai.github.io/openai-agents-python/models/)