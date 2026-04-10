import json
import re


def extract_method_name(source_code):
    """从Java源代码中提取方法名"""
    pattern = r'\b(?:public|protected|private|static|final|synchronized|abstract|transient)+\s+[\w<>\[\]]+\s+(\w+)\s*\('
    match = re.search(pattern, source_code)
    return match.group(1) if match else None


def process_json(input_file, output_file):
    # 读取整个文件内容
    with open(input_file, 'r', encoding='UTF-8`') as f:
        file_content = f.read()

    # 尝试解析为JSON数组
    try:
        data_list = json.loads(file_content)
        if not isinstance(data_list, list):
            data_list = [data_list]  # 如果是单个对象，转换为列表
    except json.JSONDecodeError:
        # 如果解析失败，尝试按行处理
        data_list = []
        for line in file_content.splitlines():
            if line.strip():  # 跳过空行
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"无法解析行: {line}")
                    continue

    output_data = []

    for data in data_list:
        if 'source_code' not in data:
            print(f"跳过缺少'source_code'字段的对象: {data}")
            continue

        source_code = data['source_code']
        method_name = extract_method_name(source_code)

        if not method_name:
            print(f"方法名提取失败: {source_code}")
            continue

        # 替换方法名（只替换第一次出现）
        marked_code = source_code.replace(method_name, "<mark>", 1)

        # 构建消息结构
        user_content = f"{marked_code}根据上面的方法体进行<mark>位置的方法名预测"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": method_name}
        ]

        output_data.append({"messages": messages})

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"成功处理 {len(output_data)} 个对象，输出到 {output_file}")


# 使用示例
if __name__ == '__main__':
    process_json('C:\\Users\\ouyangboyu\\Desktop\\EMSE-DeepCom-master\\projects\\10_folds\\fold_1.json', 'output1.json')