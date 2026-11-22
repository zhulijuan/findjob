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
    parser.add_argument("--var", action="append", default=[], metavar="KEY=VALUE", help="Prompt 模板变量，可重复传入")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--list", action="store_true", help="列出所有任务")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.list or not args.task:
        print(f"任务列表（配置: {args.config}）:")
        print(config.list_tasks())
        return

    variables = parse_vars(args.var)

    if args.stream:
        stream = run_task(config, args.task, variables, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
        print()
    else:
        content = run_task(config, args.task, variables)
        print(content)


if __name__ == "__main__":
    main()
