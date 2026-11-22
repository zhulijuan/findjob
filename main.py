import argparse

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
    parser.add_argument("--var", action="append", default=[], metavar="KEY=VALUE", help="Prompt 模板变量，可重复传入")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--list", action="store_true", help="列出所有任务")
    args = parser.parse_args()

    if args.ask and args.task:
        parser.error("--ask 与 --task 不能同时使用")

    config = load_config(args.config)

    if args.list or (not args.task and not args.ask):
        print(f"任务列表（配置: {args.config}）:")
        print(config.list_tasks())
        return

    variables = parse_vars(args.var)
    task_name = args.task
    if args.ask:
        task_name = "qa"
        variables["question"] = args.ask

    if args.stream:
        stream = run_task(config, task_name, variables, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
        print()
    else:
        response = run_task(config, task_name, variables)
        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
