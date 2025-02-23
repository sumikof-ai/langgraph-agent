from typing import Annotated
from pydantic import BaseModel, Field
from operator import add
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import javalang

# --- ノード定義 ---
class JavaMigrationState(BaseModel):
    original_code: str = Field(...,description="")
    modified_code: str = Field(default="",description="")
    changes: Annotated[list[str], add] = Field(default_factory=list, description="")
    error: str = Field(default="",description="exception message")

def parse_java_code(state: JavaMigrationState):
    """
    (1) コード解析
    """
    tree = javalang.parse.parse(state.original_code)
    modified_code = state.original_code
    return {"modified_code": modified_code}


def rule_based_conversion(state: JavaMigrationState):
    """
    (2) ルールベース変換     
    """
    # 非推奨APIリスト (例)
    DEPRECATED_APIS = [
        ("java.util.Date", "java.time.LocalDateTime"),
        ("Thread.stop", ""),

    ]
    modified_code = state.original_code

    for old_syntax,new_syntax in DEPRECATED_APIS:
        modified_code = modified_code.replace(old_syntax, new_syntax)
    return {"modified_code": modified_code}


def ai_conversion(state: JavaMigrationState):
    """
    (3) AI補助変換
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a Java code migration assistant."),
        ("human", "Convert this Java 8 code to Java 17 while preserving its behavior:\n{code}")
    ])
    llm = ChatOpenAI(model="gpt-4")
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    modified_code = chain.invoke({"code": state.modified_code})
    return {"modified_code": modified_code}


# --- LangGraph ワークフローの定義 ---
workflow = StateGraph(JavaMigrationState)
workflow.add_node("parse_java_code", RunnableLambda(parse_java_code))
workflow.add_node("rule_based_conversion", RunnableLambda(rule_based_conversion))
workflow.add_node("ai_conversion", RunnableLambda(ai_conversion))

workflow.add_edge(START, "parse_java_code")
workflow.add_edge("parse_java_code", "rule_based_conversion")
workflow.add_edge("rule_based_conversion", "ai_conversion")
workflow.add_edge("ai_conversion", END)

graph = workflow.compile()

# --- 実行例 ---
if __name__ == "__main__":
    java_code = """
    import java.util.Date;
    public class Test {
        public static void main(String[] args) {
            Date d = new Date();
            System.out.println(d);
            int num = Integer("16");
            System.out.println(num);
        }
    }
    """
    graph.get_graph().draw_ascii
    result_state = graph.invoke({"original_code": java_code})
    print("".join(result_state["modified_code"]))
