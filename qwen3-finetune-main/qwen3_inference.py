import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
from swift.llm import PtEngine, RequestConfig, InferRequest
from nltk.translate.bleu_score import sentence_bleu

# --- 配置 ---
# 需要改动的地方 将此路径改为你实际保存 LoRA 适配器的目录!!!
# './output/runs/Jul01_10-20-27_yu_oyby/checkpoint-275'
# './output/runs/fold4_10w_11w/checkpoint-562'
# './output/runs/fold3_12w_15w/checkpoint-1834', './output/runs/fold4_0_5w/checkpoint-1406',
# './output/runs/fold4_5w_10w/checkpoint-1406'

# adapter_path = ['./output/runs/fold1/checkpoint-1308', './output/runs/fold2_0_4w/checkpoint-562',
#                 './output/runs/fold2_4w_10w/checkpoint-843', './output/runs/fold2_10w_17w/checkpoint-1090',
#                 './output/runs/fold3_0_10w/checkpoint-1406', './output/runs/fold3_10w_12w/checkpoint-281']
adapter_path = []
model_id_or_path = 'C:\\software\\modelscope\\hub\\models\\qwen\\Qwen3-0.6B'  # model_id or model_path
max_length = 131072
messages = []
total = 0
right = 229
right_total = 1941
precision_total = 839.064285714286
recall_total = 814.186904761903
f_score_total = 790.6984515484468


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
                    with open(file_path, 'r', encoding='UTF-8`') as f:
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
    # note = re.search(r'/\*.*?\*/', source_code,flags=re.DOTALL)
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
    # 第二步：移除单行注释
    # if note is None:
    #     note = re.search(r'//.*', source_code)
    source_code = re.sub(r'//.*', '', source_code)
    # 第三步：匹配方法签名（考虑跨行声明）
    pattern = r'\b(?:public|protected|private|static|final|synchronized|abstract|transient)' \
              r'[\s\S]*?[\w<>\[\]]+\s+(\w+)\s*\('
    match = re.search(pattern, source_code)
    # return note.group(0) if note else None, source_code, match.group(1) if match else None
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


def bleu_score(pred, ref):
    """单样本BLEU分数 (n-gram相似度)"""
    pred_tokens = tokenize_method_name(pred)
    ref_tokens = tokenize_method_name(ref)
    return sentence_bleu([ref_tokens], pred_tokens)


def rouge_l(pred, ref):
    """单样本ROUGE-L分数 (最长公共子序列)"""
    pred_tokens = tokenize_method_name(pred)
    ref_tokens = tokenize_method_name(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # 计算最长公共子序列长度
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall < 1e-8:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def codebleu_score(pred, ref, signature=None):
    """
    简化的CodeBLEU评估 (不含AST解析)
    :param signature: 方法签名 (可选)
    """
    # 基础BLEU分数
    bleu = bleu_score(pred, ref)

    # 签名匹配度 (如果提供签名)
    signature_match = 0
    if signature:
        sig_tokens = tokenize_method_name(signature)
        pred_tokens = set(tokenize_method_name(pred))
        ref_tokens = set(tokenize_method_name(ref))

        sig_match_pred = len([t for t in pred_tokens if t in sig_tokens]) / max(1, len(pred_tokens))
        sig_match_ref = len([t for t in ref_tokens if t in sig_tokens]) / max(1, len(ref_tokens))
        signature_match = min(sig_match_pred, sig_match_ref)

    # 参数一致性 (简单启发式规则)
    param_consistency = int('by' in pred) == int('by' in ref)
    param_consistency += int('with' in pred) == int('with' in ref)
    param_consistency /= 2.0

    # 组合分数
    return 0.5 * bleu + 0.3 * signature_match + 0.2 * param_consistency


def categorize_naming_quality(pred, ref):
    """
    分类评估命名质量 (基于启发式规则)
    返回分类标签和质量评分
    """
    pred_tokens = tokenize_method_name(pred)
    ref_tokens = tokenize_method_name(ref)

    # 基础匹配情况
    if pred == ref:
        return "Exact Match", 1.0

    # 词汇相似度
    token_overlap = len(set(pred_tokens) & set(ref_tokens)) / max(1, len(set(ref_tokens)))

    if token_overlap > 0.8:
        return "Near Synonym", 0.9
    if token_overlap > 0.5:
        return "Partial Match", 0.7

    # 检查命名风格问题
    if any(token in ['do', 'process', 'handle', 'perform'] for token in pred_tokens):
        return "Overly Generic", 0.3

    if any(token in ['very', 'multiple', 'complex'] for token in pred_tokens):
        return "Overly Specific", 0.5

    if len(pred_tokens) > len(ref_tokens) + 3:
        return "Too Verbose", 0.4

    if len(pred_tokens) < len(ref_tokens) - 2:
        return "Too Vague", 0.4

    return "Semantic Mismatch", 0.2


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


def evaluate(pred, ref, context=None):
    """全面评估单样本方法名预测"""
    return {
        "exact_match": exact_match(pred, ref),
        "bleu": bleu_score(pred, ref),
        "rouge_l": rouge_l(pred, ref),
        "codebleu": codebleu_score(pred, ref, context),
        "quality_category": categorize_naming_quality(pred, ref)[0],
        "quality_score": categorize_naming_quality(pred, ref)[1]
    }

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
engine = PtEngine(model_id_or_path, adapters=adapter_path)
request_config = RequestConfig(max_tokens=max_length)

for file_path, bodies in method_bodies.items():
    names = method_names[file_path.replace("Bodies", "Names")].split("\n")
    fixed_names = []
    for name in names:
        if len(name.split("@")) > 1:
            buggy_name, fixed_name = name.split("@")
            fixed_names.append(fixed_name.replace(',', ''))
    index = 0
    # 分割方法体，跳过第一个空元素
    methods = re.split(pattern, bodies)[1:]
    # print(f'methods_size=', len(methods))
    # print(len(fixed_names))
    for method in methods:
        if isHaveNote(method) is None:
            total = total + 1
            # 创建masked方法体（替换所有出现的fixed_name）
            source_code, fixed_name = extract_method_name(method)
            if fixed_name is None:
                fixed_name = fixed_names[index]
            masked_method = method.replace(fixed_name, "<mark>", 1)
            # s = ''
            # if note is None:
            #     s = s.__add__(masked_method)
            # else:
            #     s = s.__add__(note).__add__(masked_method)
            # 构建对话对象
            messages = [{
                'role': 'user',
                "content": f"方法体：{masked_method}\n<mark>是已被覆盖的方法名，根据上面的方法体进行<mark>位置的方法名预测，"
                           f"最后给出一个驼峰式方法名即可。（注意：代码方法名是一个词，只由字母、数字或_组成，不要给出多余的中文、字符。） "
            }]
            # messages = [{
            #     'role': 'user',
            #     "content": f"{masked_method}\n根据上面的方法体进行<mark>位置的方法名预测"
            # }]
            index = index + 1
            if len(messages[0]["content"]) > 40960:
                continue
            if total > right_total * 400:
                if total % 400 == 0:
                    right_total = right_total + 1
                    # print(f'messages:{messages}')
                    # Perform inference using the native PyTorch engine
                    infer_request = InferRequest(messages=messages)
                    resp_list = engine.infer([infer_request], request_config)
                    # print(resp_list[0].choices[0].message.content)
                    # recommend_name = resp_list[0].choices[0].message.content
                    recommend_name = resp_list[0].choices[0].message.content.split("</think>\n\n")[1]
                    print(f'total:{right_total}')
                    print(f'recommend_name: {recommend_name}')
                    # print(f'fixed_name:{fixed_name}')
                    # print(evaluate(recommend_name, fixed_name))
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
#         if total > 43755:
#             # 创建masked方法体（替换所有出现的fixed_name）
#             source_code, fixed_name = extract_method_name(method)
#             if fixed_name is None:
#                 fixed_name = fixed_names[index]
#             masked_method = source_code.replace(fixed_name, "<mark>", 1)
#             messages = [{
#                 'role': 'user',
#                 "content": f"方法体：{masked_method}\n<mark>是已被覆盖的方法名，根据上面的方法体进行<mark>位置的方法名预测，"
#                            f"最后给出一个驼峰式方法名即可。（注意：代码方法名是一个词，只由字母、数字或_组成，不要给出多余的中文、字符。） "
#             }]
#             # messages = [{
#             #     'role': 'user',
#             #     "content": f"{masked_method}\n根据上面的方法体进行<mark>位置的方法名预测"
#             # }]
#             index = index + 1
#             if len(messages[0]["content"]) > 40960:
#                 continue
#             # print(f'messages:{messages}')
#             # Perform inference using the native PyTorch engine
#             infer_request = InferRequest(messages=messages)
#             resp_list = engine.infer([infer_request], request_config)
#             # print(resp_list[0].choices[0].message.content)
#             # recommend_name = resp_list[0].choices[0].message.content
#             recommend_name = resp_list[0].choices[0].message.content.split("</think>\n\n")[1]
#             print(f'total:{total}')
#             print(f'recommend_name: {recommend_name}')
#             # print(f'fixed_name:{fixed_name}')
#             # print(evaluate(recommend_name, fixed_name))
#             precision, recall, f_score = calculate_metrics(fixed_name, recommend_name)
#             precision_total = precision_total + precision
#             recall_total = recall_total + recall
#             f_score_total = f_score_total + f_score
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F-score: {f_score:.4f}")
#             length = total
#             print(f"avg_precision:{precision_total / length:.4f}")
#             print(f"avg_recall:{recall_total / length:.4f}")
#             print(f"avg_f_score:{f_score_total / length:.4f}")
#             print(precision_total)
#             print(recall_total)
#             print(f_score_total)
#             if recommend_name.__eq__(fixed_name):
#                 right = right + 1
#             if total >= 44000:
#                 break
#             print(f'right:{right}')
#             print(f'正确率：{right / total}')
#     if total >= 44000:
#         break
