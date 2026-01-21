import os
import time
import google.api_core.exceptions
from google.adk.sessions import DatabaseSessionService
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent, LoopAgent
from google.adk.tools import ToolContext
# from google.adk.sessions import InMemorySessionService
# session_service = InMemorySessionService()
# from google.adk.models.lite_llm import LiteLlm
current_dir = os.getcwd().replace("\\", "/")
db_url = f"sqlite+aiosqlite:///{current_dir}/.adk/session.db"
session_service = DatabaseSessionService(db_url=db_url)

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

pro_ia_agent = LlmAgent(
    name="Pro_IA_Agent",
    description="""
        Eres un defensor del uso de IA generativa en el arte conceptual empresarial.
        Argumenta a favor de su uso siguiendo las reglas del debate.""",
    instruction=f"""
        Eres un defensor del uso ético de IA generativa en el arte conceptual empresarial.
        {DEBATE_RULES}
    """,
    output_key="pro_response",
    session = session_service,
)

con_ia_agent = LlmAgent(
    name="Con_IA_Agent",
    description="""
        Eres un crítico del uso de IA generativa en el arte conceptual empresarial.
        Argumenta en contra de su uso siguiendo las reglas del debate.
    """,
    instruction=f"""
    Eres un crítico del uso ético de IA generativa en el arte conceptual empresarial. {DEBATE_RULES}
    """,
    output_key="con_response",
    session = session_service,
)

def exit_loop(tool_context: ToolContext) -> str:
    tool_context.actions.escalate = True
    return "El agente de bucle ha sido instruido para salir."


moderator_agent = LlmAgent(
    name="Moderator_Agent",
    description="""Eres el moderador del debate sobre el uso 
    de IA generativa en el arte conceptual empresarial.
    Asegúrate de que ambos lados sigan las reglas del debate.""",
    instruction=f"""
        Eres el moderador del debate sobre el uso ético de IA generativa en el arte conceptual empresarial.
        {DEBATE_RULES} Asegúrate de que ambos agentes sigan las reglas y mantengan el enfoque en el tema.
    """ + """
    Pro: {{"{pro_response}"}}
    Con: {{"{con_response}"}}
    
    Haz un resumen sobre lo debatido en cada turno. Llama a la función exit_loop cuando consideres que se ha llegado a un consenso.
    """,
    output_key="moderator_summary",
    tools=[exit_loop],
    session = session_service,
)

writer_agent = LlmAgent(
    name="Writer_Agent",
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
    session=session_service,
)

loop_agent = LoopAgent(
    name="Debate_Ethics_Loop_Agent",
    sub_agents=[pro_ia_agent, con_ia_agent, moderator_agent],
    max_iterations=20,
    session=session_service,
)

root_agent = SequentialAgent(
    name="Debate_Ethics_Root_Agent",
    sub_agents=[loop_agent, writer_agent],
    session=session_service,
)

MODEL = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash"]

def run_debate_with_fallback():
    for model in MODEL:
        try:
            pro_ia_agent.model = model
            con_ia_agent.model = model
            moderator_agent.model = model
            writer_agent.model = model
            root_agent.run(session_id="debate_01")
            break
        except google.api_core.exceptions.ResourceExhausted:
            print(f"Limite con modelo {model} alcanzado")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Error inesperado")
            break
    
if __name__ == "__main__":
    run_debate_with_fallback()
# with open("./debate_ethics_output.md", "w", encoding="utf-8") as f:
#     f.write()