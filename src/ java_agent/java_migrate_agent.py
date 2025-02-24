from typing import Annotated, Any
from pydantic import BaseModel, Field
import operator
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI
import javalang

from nodes.passive_goal_creator import Goal, PassiveGoalCreator
from nodes.response_optimizer import ResponseOptimizer
from nodes.query_decomposer import DecomposedTasks,QueryDecomposer
from nodes.task_executer import TaskExecutor
from nodes.result_aggregator import ResultAggregator

# --- ノード定義 ---
class JavaMigrationState(BaseModel):
    original_code: str = Field(...,description="")
    modified_code: str = Field(default="",description="")

    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス定義")
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")
    error: str = Field(default="",description="exception message")

class Java17ConverterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.graph = self.build_graph()


    def build_graph(self):
        """
        # --- LangGraph ワークフローの定義 ---
        """
        workflow = StateGraph(JavaMigrationState)
        workflow.add_node("parse_java_code", RunnableLambda(self.parse_java_code))
        workflow.add_node("rule_based_conversion", RunnableLambda(self.rule_based_conversion))
        workflow.add_node("goal_setting", RunnableLambda(self._goal_setting))
        workflow.add_node("decompose_query", RunnableLambda(self._decompose_query))
        workflow.add_node("execute_task", RunnableLambda(self._execute_task))
        workflow.add_node("aggregate_results", RunnableLambda(self._aggregate_results))

        workflow.add_edge(START, "parse_java_code")
        workflow.add_edge("parse_java_code", "rule_based_conversion")
        workflow.add_edge("rule_based_conversion", "goal_setting")
        workflow.add_edge("goal_setting", "decompose_query")
        workflow.add_edge("decompose_query", "execute_task")
        workflow.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "execute_task", False: "aggregate_results"},
        )
        workflow.add_edge("aggregate_results", END)

        graph = workflow.compile()
        return graph

    def run(self, java_code :str):
        result = self.graph.invoke({"original_code": java_code})
        return result

    def parse_java_code(self, state: JavaMigrationState):
        """
        コード解析
        """
        tree = javalang.parse.parse(state.original_code)
        print(tree)
        modified_code = state.original_code
        return {"modified_code": modified_code}


    def rule_based_conversion(self, state: JavaMigrationState):
        """
        ルールベース変換     
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

    def _goal_setting(self, state: JavaMigrationState):
        """
        入力を最適化された目標に変換
        """
        goal_creator = PassiveGoalCreator(self.llm)
        goal: Goal = goal_creator.run("次の Java 8 プログラムを Java 17 "
                         f"プログラムに変換して下さい。:\n{state.modified_code}")
        optimized_goal = goal.text

        response_optimizer = ResponseOptimizer(self.llm)
        optimized_response = response_optimizer.run(optimized_goal)
        return {
            "optimized_goal": optimized_goal,
            "optimized_response": optimized_response,
        }
    
    def _decompose_query(self, state: JavaMigrationState) -> dict[str, Any]:
        """
        最適化された目標を元にタスクを細分化
        """
        query_decomposer = QueryDecomposer(self.llm)
        decomposed_tasks: DecomposedTasks = query_decomposer.run(
            query=state.optimized_goal
        )
        return {"tasks": decomposed_tasks.values}
    
    def _execute_task(self, state: JavaMigrationState) -> dict[str, Any]:
        """
        各タスクを実行
        """
        current_task = state.tasks[state.current_task_index]
        task_executor = TaskExecutor(self.llm)
        result = task_executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(
            self, state: JavaMigrationState
        ) -> dict[str, Any]:
        """
        最終出力の編集
        """
        result_aggregator = ResultAggregator(self.llm)
        final_output = result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results,
        )
        return {"final_output": final_output}


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

    llm = ChatOpenAI(model="gpt-4")

    agent = Java17ConverterAgent(llm=llm)
    # print(agent.graph.get_graph().draw_ascii())
    result_state = agent.run(java_code=java_code)
    print(result_state)
    print("".join(result_state["final_output"]))
