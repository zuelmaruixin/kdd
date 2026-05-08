import pandas as pd
from pathlib import Path


NUMERIC_TOLERANCE = 1.0e-2


def normalize_cell(value, *, numeric_tolerance=NUMERIC_TOLERANCE):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "__EMPTY__"
    try:
        numeric = float(text)
    except ValueError:
        return text.lower()
    if numeric_tolerance > 0:
        numeric = round(numeric / numeric_tolerance) * numeric_tolerance
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.12g}"


def load_csv_to_unordered_columns(csv_path):
    """读取 CSV，抛弃列名，把每一列变成一个排好序的元组（无序向量）"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    columns_data = []
    for col in df.columns:
        # 把列里面的数据转成字符串，去除空格，过滤空值，然后排序
        col_values = sorted([normalize_cell(x) for x in df[col].dropna()])
        columns_data.append(tuple(col_values))
    return columns_data


def evaluate_batch(run_id):
    run_dir = Path(f"artifacts/runs/{run_id}")

    if not run_dir.exists():
        print(f"❌ 找不到运行记录文件夹: {run_dir}")
        return

    total_evaluated = 0
    correct_count = 0
    failed_tasks = []

    print(f"🔍 正在扫描 {run_dir} 下的测试结果...\n")

    # 遍历该 run_id 下的所有 task_xxx 文件夹
    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        task_id = task_dir.name
        gold_path = Path(f"data/public/output/{task_id}/gold.csv")
        pred_path = task_dir / "prediction.csv"

        # 如果没有标准答案，跳过（针对 hidden test）
        if not gold_path.exists():
            continue

        total_evaluated += 1

        # 没生成预测文件，直接 0 分
        if not pred_path.exists():
            print(f"[{task_id}] ❌ 0分 (Agent 崩溃或超时未交卷)")
            failed_tasks.append(task_id)
            continue

        gold_cols = load_csv_to_unordered_columns(gold_path)
        pred_cols = load_csv_to_unordered_columns(pred_path)

        # 列向量匹配判分逻辑
        is_correct = True
        for g_col in gold_cols:
            if g_col not in pred_cols:
                is_correct = False
                break

        if is_correct:
            print(f"[{task_id}] ✅ 1分")
            correct_count += 1
        else:
            print(f"[{task_id}] ❌ 0分 (答案数据不匹配)")
            failed_tasks.append(task_id)

    # 打印最终成绩单
    print("\n" + "=" * 45)
    print(f"📊 批量评测成绩单 | Run ID: {run_id}")
    print("=" * 45)
    print(f"总计评测任务 : {total_evaluated}")
    print(f"正确 (1分)   : {correct_count}")
    print(f"错误 (0分)   : {total_evaluated - correct_count}")

    if total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        print(f"🏆 整体准确率 : {accuracy:.2f}%")
    print("=" * 45)

    # 贴心地把错题列出来，方便你后续去查 trace.json
    if failed_tasks:
        print(f"\n💡 错题本 (建议重点查看它们的 trace.json):")
        print(", ".join(failed_tasks))


if __name__ == "__main__":
    # TODO: 跑完数据后，去 artifacts/runs/ 下看一眼最新生成的文件夹名字，填到这里
    RUN_ID = ""

    evaluate_batch(RUN_ID)
