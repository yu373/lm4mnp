from openai import OpenAI, APIStatusError, InternalServerError
import time
import sys

key = ''
client = OpenAI(api_key=key, base_url="")

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re

messages = []
total = 0
right = 0
right_total = 0
precision_total = 0
recall_total = 0
f_score_total = 0


def read_files(folder_path):
    method_bodies = {}
    method_names = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith('readme.txt') \
                    and not file.lower().endswith("projects.txt") \
                    and not file.lower().endswith("methodinfos.txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='UTF-8') as f:  # 修正了编码参数
                        content = f.read()
                    if file.lower().endswith("methodbodies.txt"):
                        method_bodies[file_path] = content
                    if file.lower().endswith("methodnames.txt"):
                        method_names[file_path] = content
                except Exception as e:
                    print(f"无法读取文件 {file_path} : {e}")
    return method_bodies, method_names


def extract_method_name(source_code):
    """从Java源代码中提取方法名（自动忽略多行/单行注释）"""
    # 第一步：移除多行注释（包括可能跨行的注释）
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
    # 第二步：移除单行注释
    source_code = re.sub(r'//.*', '', source_code)
    # 第三步：匹配方法签名（考虑跨行声明）
    pattern = r'\b(?:public|protected|private|static|final|synchronized|abstract|transient)' \
              r'[\s\S]*?[\w<>\[\]]+\s+(\w+)\s*\('
    match = re.search(pattern, source_code)
    return source_code, match.group(1) if match else None


def exact_match(pred, ref):
    """单样本精确匹配评估"""
    return int(pred == ref)


def tokenize_method_name(name):
    """
    将方法名拆分为token列表（基于camel case和下划线），并转换为小写。
    参数:
        name (str): 方法名（如 "getMenuList"）
    返回:
        list: token列表（如 ["get", "menu", "list"]）
    """
    # 使用正则表达式拆分camel case和下划线
    tokens = re.split(r'(?<=[a-z])(?=[A-Z])|_', name)
    # 过滤空字符串并转换为小写
    tokens = [token.lower() for token in tokens if token]
    return tokens


def calculate_metrics(oracle_name, recommended_name):
    """
    计算precision、recall和F-score。
    参数:
        oracle_name (str): 真实方法名
        recommended_name (str): 推荐方法名
    返回:
        tuple: (precision, recall, f_score)
    """
    # 拆分token
    tokens_oracle = set(tokenize_method_name(oracle_name))
    tokens_recommended = set(tokenize_method_name(recommended_name))

    # 计算交集
    intersection = tokens_recommended & tokens_oracle
    num_intersection = len(intersection)

    # 处理除零错误
    num_recommended = len(tokens_recommended) if tokens_recommended else 1e-9
    num_oracle = len(tokens_oracle) if tokens_oracle else 1e-9

    # 计算precision和recall
    precision = num_intersection / num_recommended
    recall = num_intersection / num_oracle

    # 计算F-score（避免分母为零）
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def call_api_with_retry(masked_method, max_retries=5, initial_delay=1):
    """
    调用API并带有重试机制
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": "You are an expert in Java method name prediction."},
                    {"role": "user", "content": f"方法体：{masked_method}\n<mark>是已被覆盖的方法名，根据上面的方法体进行<mark>位置的方法名预测，"
                                                f"最后给出一个驼峰式方法名即可。（注意：代码方法名是一个词，只由字母、数字或_组成，不要给出多余的中文、字符。） "}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except InternalServerError as e:
            if e.status_code == 504 and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  # 指数退避
                print(f"API返回404错误，第{attempt + 1}次重试，等待{delay}秒...")
                time.sleep(delay)
            else:
                raise e
        except APIStatusError as e:
            if e.status_code == 404 and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  # 指数退避
                print(f"API返回404错误，第{attempt + 1}次重试，等待{delay}秒...")
                time.sleep(delay)
            else:
                raise e
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"API调用出错: {e}，第{attempt + 1}次重试，等待{delay}秒...")
                time.sleep(delay)
            else:
                raise e

    # 如果所有重试都失败了，返回一个默认值或抛出异常
    return "<api_error>"

def isHaveNote(source_code):
    match = re.search(r'/\*.*?\*/', source_code)
    if match:
        return match.group(0)
    else:
        return None
    match = re.search(r'//.*', '', source_code)
    return match.group(0) if match else None


method_bodies, method_names = read_files("C:\\Users\\ouyangboyu\\Desktop\\BenMark\\dataset")
pattern = r"#METHOD_BODY_\d+#=+"
results = []  # 存储所有JSON对象的列表

for file_path, bodies in method_bodies.items():
    names = method_names[file_path.replace("Bodies", "Names")].split("\n")
    fixed_names = []
    for name in names:
        if len(name.split("@")) > 1:
            buggy_name, fixed_name = name.split("@")
            fixed_names.append(fixed_name.replace(',', ''))
    # 分割方法体，跳过第一个空元素
    methods = re.split(pattern, bodies)[1:]
    index = 0
    for method in methods:
        if isHaveNote(method) is None:
            total = total + 1
            # 创建masked方法体（替换所有出现的fixed_name）
            source_code, fixed_name = extract_method_name(method)
            if fixed_name is None:
                fixed_name = fixed_names[index]
            masked_method = source_code.replace(fixed_name, "<mark>", 1)
            if total > right_total * 400:
                if total % 400 == 0:
                    right_total = right_total + 1
                    recommend_name = call_api_with_retry(masked_method)
                    index = index + 1
                    print(f'total:{right_total}')
                    print(f'recommend_name: {recommend_name}')

                    precision, recall, f_score = calculate_metrics(fixed_name, recommend_name)
                    precision_total = precision_total + precision
                    recall_total = recall_total + recall
                    f_score_total = f_score_total + f_score
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F-score: {f_score:.4f}")
                    length = right_total
                    print(f"avg_precision:{precision_total / length:.4f}")
                    print(f"avg_recall:{recall_total / length:.4f}")
                    print(f"avg_f_score:{f_score_total / length:.4f}")
                    print(precision_total)
                    print(recall_total)
                    print(f_score_total)

                    if recommend_name.__eq__(fixed_name):
                        right = right + 1

                    print(f'right:{right}')
                    print(f'正确率：{right / right_total}')


# for file_path, bodies in method_bodies.items():
#     names = method_names[file_path.replace("Bodies", "Names")].split("\n")
#     fixed_names = []
#     for name in names:
#         if len(name.split("@")) > 1:
#             buggy_name, fixed_name = name.split("@")
#             fixed_names.append(fixed_name.replace(',', ''))
#     # 分割方法体，跳过第一个空元素
#     methods = re.split(pattern, bodies)[1:]
#     index = 0
#     for method in methods:
#         total = total + 1
#         if total > 39000:
#             # 创建masked方法体（替换所有出现的fixed_name）
#             source_code, fixed_name = extract_method_name(method)
#             if fixed_name is None:
#                 fixed_name = fixed_names[index]
#             masked_method = source_code.replace(fixed_name, "<mark>", 1)
#
#             # 使用带重试机制的API调用
#             try:
#                 recommend_name = call_api_with_retry(masked_method)
#                 index = index + 1
#                 print(f'total:{total}')
#                 print(f'recommend_name: {recommend_name}')
#
#                 precision, recall, f_score = calculate_metrics(fixed_name, recommend_name)
#                 precision_total = precision_total + precision
#                 recall_total = recall_total + recall
#                 f_score_total = f_score_total + f_score
#                 print(f"Precision: {precision:.4f}")
#                 print(f"Recall: {recall:.4f}")
#                 print(f"F-score: {f_score:.4f}")
#                 length = total
#                 print(f"avg_precision:{precision_total / length:.4f}")
#                 print(f"avg_recall:{recall_total / length:.4f}")
#                 print(f"avg_f_score:{f_score_total / length:.4f}")
#                 print(precision_total)
#                 print(recall_total)
#                 print(f_score_total)
#
#                 if recommend_name.__eq__(fixed_name):
#                     right = right + 1
#
#                 print(f'right:{right}')
#                 print(f'正确率：{right / total}')
#
#             except Exception as e:
#                 print(f"处理第{total}个方法时出错: {e}")
#                 # 可以选择跳过这个样本或者终止程序
#                 sys.exit()
#
#         if total >= 44000:
#             break
#     if total >= 44000:
#         break