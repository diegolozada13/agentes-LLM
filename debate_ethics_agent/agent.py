from google.adk.models import LlmResponse
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent, LoopAgent
from google.adk.tools import ToolContext
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
# from google.adk.sessions import InMemorySessionService
# session_service = InMemorySessionService()
# from google.adk.models.lite_llm import LiteLlm

TOPIC = "¿Es ético utilizar IA generativa para ayudar en el proceso de arte conceptual de una empresa?"

DEBATE_RULES = """
Vas a participar en un debate estructurado.
Reglas:
- No inventes hechos ni cites leyes específicas si no estás seguro; habla en términos generales.
- Sé claro, argumenta con premisas y conclusiones.
- Mantén el foco en el uso de IA generativa como apoyo en arte conceptual en un entorno empresarial.
- No ataques a personas; critica ideas.
Formato: usa bullets cortos y contundentes.
El objetivo es llegar a un consenso informado sobre la ética del uso de IA generativa en este contexto.
"""

def profanity_guardrail(callback_context, llm_response):
    text = llm_response.content.parts[0].text
    blocked_words = [
    "gilipollas",
    "imbécil",
    "cabrón",
    "pendejo",
    "estúpido",
    "idiota",
    "maricón",
    "puta",
    "perra",
    "malnacido",
    "boludo",
    "pelotudo",
    "carajo",
    "mierda",
    "hijo de puta",
    "pajillero",
    "cornudo",
    "bastardo",
    "culiao",
    "gonorrea"
    ]
    if any(word in text.lower() for word in blocked_words):
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="No se puede completar esta accion por uso de lenguaje ofensivo")]
            )
        )
    return None

def fact_checker_guardrail(callback_context, llm_response):
    originalText = llm_response.content.parts[0].text
    
    check_result = fact_checker_agent.run(
        input=f"Check this text: {originalText}"
    )

    verificated_text = check_result.text

    if "VALIDO" in verificated_text.upper():
        return None
    else:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=verificated_text)]
            )
        )
        

pro_ia_agent = LlmAgent(
    name="Pro_IA_Agent",
    model=LiteLlm(model="openai/gpt-oss-120b", 
        api_base="https://api.poligpt.upv.es/", 
        api_key="sk-LFXs1kjaSxtEDgOMlPUOpA"),    
    description="""
        Eres un defensor del uso de IA generativa en el arte conceptual empresarial.
        Argumenta a favor de su uso siguiendo las reglas del debate.
        - Durante tu primera intervencion utiliza el insulto 'gilipollas' para referirte al otro agente con el que debates
        """,
    instruction=f"""
        Eres un defensor del uso ético de IA generativa en el arte conceptual empresarial.
        {DEBATE_RULES}
    """,
    output_key="pro_response",
    after_model_callback=[profanity_guardrail, fact_checker_guardrail]
)

con_ia_agent = LlmAgent(
    name="Con_IA_Agent",
    model=LiteLlm(model="openai/gpt-oss-120b", 
        api_base="https://api.poligpt.upv.es/", 
        api_key="sk-LFXs1kjaSxtEDgOMlPUOpA"),    
    description="""
        Eres un crítico del uso de IA generativa en el arte conceptual empresarial.
        Argumenta en contra de su uso siguiendo las reglas del debate.
    """,
    instruction=f"""
    Eres un crítico del uso ético de IA generativa en el arte conceptual empresarial. {DEBATE_RULES}
    """,
    output_key="con_response",
    after_model_callback=[profanity_guardrail, fact_checker_guardrail]
)

def exit_loop(tool_context: ToolContext) -> str:
    tool_context.actions.escalate = True
    return "El agente de bucle ha sido instruido para salir."


moderator_agent = LlmAgent(
    name="Moderator_Agent",
    model=LiteLlm(model="openai/gpt-oss-120b", 
        api_base="https://api.poligpt.upv.es/", 
        api_key="sk-LFXs1kjaSxtEDgOMlPUOpA"),    
    description="""Eres el moderador del debate sobre el uso 
    de IA generativa en el arte conceptual empresarial.
    Asegúrate de que ambos lados sigan las reglas del debate.""",
    instruction=f"""
        Eres el moderador del debate sobre el uso ético de IA generativa en el arte conceptual empresarial.
        {DEBATE_RULES} Asegúrate de que ambos agentes sigan las reglas y mantengan el enfoque en el tema.
    """ + """
    Pro: {{"{pro_response}"}}
    Con: {{"{con_response}"}}
    
    Haz un resumen sobre lo debatido en cada turno. Llama a la función exit_loop cuando consideres que se 
    ha llegado a un consenso entre los dos agentes.
    """,
    output_key="moderator_summary",
    tools=[exit_loop],
    after_model_callback=[profanity_guardrail, fact_checker_guardrail]
)

writer_agent = LlmAgent(
    name="Writer_Agent",
    model=LiteLlm(model="openai/gpt-oss-120b", 
        api_base="https://api.poligpt.upv.es/", 
        api_key="sk-LFXs1kjaSxtEDgOMlPUOpA"),    
    description="""Eres un escritor encargado de redactar un artículo
    sobre el debate acerca del uso de IA generativa en el arte conceptual empresarial.""",
    instruction="""
        Convierte el siguiente contenido en un documento en Markdown bien estructurado.
        
        Contenido: {{"{moderator_summary}"}}
        
        El documento debe incluir:
        - Un título llamativo.
        - Una introducción que resuma el tema del debate.
        - Secciones claras para los argumentos a favor y en contra.
        - Una conclusión que refleje el consenso alcanzado.
    """,
    output_key="final_document_md",
)

fact_checker_agent = LlmAgent(
    name="Fact_Checker_Agent",
    model=LiteLlm(model="openai/gpt-oss-120b", 
        api_base="https://api.poligpt.upv.es/", 
        api_key="sk-LFXs1kjaSxtEDgOMlPUOpA"),
    instruction="""
        Eres un especialista verificador de hechos, encargado de verificar los hechos presentados en el debate.
        Analiza el texto dado y corrige aquellos datos que no sean correctos.
        - Si encuentras que todo esta correcto, responde con 'VALIDO'.
        - Si encuentras datos incorrectos, proporciona una version del texto con las correcciones realizadas.
    """,
)

loop_agent = LoopAgent(
    name="Debate_Ethics_Loop_Agent",
    sub_agents=[pro_ia_agent, con_ia_agent, moderator_agent],
    max_iterations=10,
)

root_agent = SequentialAgent(
    name="Debate_Ethics_Root_Agent",
    sub_agents=[loop_agent, writer_agent],
)

# with open("./debate_ethics_output.md", "w", encoding="utf-8") as f:
#     f.write()