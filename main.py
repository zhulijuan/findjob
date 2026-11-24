import argparse
from pathlib import Path

from harness.config import load_config
from harness.runner import run_task


def parse_vars(items: list[str]) -> dict[str, str]:
    variables = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--var 格式应为 KEY=VALUE，收到: {item}")
        variables[key] = value
    return variables


def main():
    parser = argparse.ArgumentParser(description="配置驱动的 LLM 任务 harness")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--task", help="要运行的任务名（用 --list 查看）")
    parser.add_argument("--ask", help="直接向大模型提问，走配置中的 qa 任务")
    parser.add_argument("--rag", help="基于信贷知识库问答（需先 --build-index）")
    parser.add_argument("--build-index", action="store_true", help="从语料目录重建向量索引")
    parser.add_argument(
        "--eval", nargs="?", const="data/eval/retrieval_testset.jsonl", metavar="TESTSET",
        help="跑检索对比评测（4 配置 × recall/MRR），默认测试集 data/eval/retrieval_testset.jsonl",
    )
    parser.add_argument("--note", help="本次评测变更备注（配合 --eval 使用，写进报告的变更说明）")
    parser.add_argument("--var", action="append", default=[], metavar="KEY=VALUE", help="Prompt 模板变量，可重复传入")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--list", action="store_true", help="列出所有任务")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.ask, args.task, args.rag, args.build_index, args.eval)) > 1:
        parser.error("--ask/--task/--rag/--build-index/--eval 一次只能用一个")

    config = load_config(args.config)

    if args.build_index:
        from harness.rag import build_index

        print(f"索引构建完成，共 {build_index(config)} 块")
        return

    if args.eval is not None:
        from harness.eval import evaluate

        result = evaluate(config, args.eval, note=args.note)
        print(Path(result["markdown"]).read_text(encoding="utf-8"))
        print(f"明细: {result['jsonl']}")
        return

    if args.list or (not args.task and not args.ask and not args.rag):
        print(f"任务列表（配置: {args.config}）:")
        print(config.list_tasks())
        return

    variables = parse_vars(args.var)
    task_name = args.task
    if args.ask:
        task_name = "qa"
        variables["question"] = args.ask

    if args.rag:
        from harness.rag import answer_question

        response, hits = answer_question(config, args.rag, stream=args.stream)
    elif args.stream:
        stream = run_task(config, task_name, variables, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
        print()
        return
    else:
        response = run_task(config, task_name, variables)
        print(response.choices[0].message.content)
        return

    if args.stream:
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
        print()
    else:
        print(response.choices[0].message.content)
    print("\n引用来源:")
    for rank, (score, hit) in enumerate(hits, 1):
        heading = hit["heading"] or "正文"
        print(f"  [{rank}] {hit['source_file']}｜{heading}｜score={score:.3f}")


if __name__ == "__main__":
    main()
