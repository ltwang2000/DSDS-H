import os
import shutil
import sys
from pathlib import Path


def smart_move(list_file, source_dir, target_dir):
    """智能文件移动，处理各种特殊情况"""
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(exist_ok=True)

    # 预加载所有源文件名（小写化加速匹配）
    source_files = {f.name.lower(): f for f in source.iterdir()}

    moved = 0
    missing = 0

    with open(list_file, 'r') as f:
        for line in f:
            raw_name = line.strip()
            if not raw_name:
                continue

            # 生成可能的文件名变体
            candidates = {
                raw_name,
                raw_name + '.jpg',
                raw_name + '.jpeg',
                raw_name + '.png',
                raw_name.split('.')[0],  # 去掉扩展名
                raw_name.replace('_', ' '),
                raw_name.replace(' ', '_')
            }

            # 查找匹配
            matched = False
            for name in candidates:
                lower_name = name.lower()
                if lower_name in source_files:
                    src = source_files[lower_name]
                    dst = target / src.name
                    try:
                        shutil.move(str(src), str(dst))
                        print(f"\033[32m✓ MOVED: {src.name}\033[0m")
                        moved += 1
                        matched = True
                        break
                    except Exception as e:
                        print(f"\033[31m! ERROR: {src.name} → {e}\033[0m")

            if not matched:
                print(f"\033[33m? MISSING: {raw_name}\033[0m")
                missing += 1

    print(f"\nRESULTS: Total {moved + missing} | Moved: {moved} | Missing: {missing}")
    if missing > 0:
        print("\nTROUBLESHOOTING TIPS:")
        print("1. 检查文件名大小写是否完全匹配")
        print("2. 确认列表文件中的文件名是否包含正确扩展名")
        print("3. 执行以下命令对比差异:")
        print(f"   comm -23 <(sort {list_file}) <(ls {source_dir} | sed 's/\\..*$//' | sort)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python smart_move.py <list_file> <source_dir> <target_dir>")
        print("Example: python smart_move.py valid.txt train test_2016_val")
        sys.exit(1)

    smart_move(sys.argv[1], sys.argv[2], sys.argv[3])