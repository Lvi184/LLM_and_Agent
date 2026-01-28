def analyze_separators(text):
    """
    分析文本中的分割符号及其出现次数:字符串分析
    
    Args:
        text (str): 待分析的目标文本
    
    Returns:
        dict: 键为分割符号，值为出现次数
    """
    # 定义常见的分割符号集合（你可以根据需求扩展）
    separators = {
        ' ', '\n', '\t', '\r', ',', '，', '.', '。', '!', '！', 
        '?', '？', ';', '；', ':', '：', '-', '——', '_', '—',
        '(', ')', '（', '）', '[', ']', '【', '】', '{', '}', 
        '<', '>', '《', '》', '/', '\\', '|', '、', '·', '`',
        '"', '“', '”', '\'', '‘', '’', '@', '#', '$', '%',
        '^', '&', '*', '+', '=', '~', '￥', '…', '·'
    }
    
    # 初始化统计字典
    separator_count = {}
    
    # 遍历文本中的每个字符
    for char in text:
        if char in separators:
            # 如果字符是分割符，更新统计
            if char in separator_count:
                separator_count[char] += 1
            else:
                separator_count[char] = 1
    
    return separator_count

def print_separator_stats(separator_count):
    """
    格式化打印分割符号统计结果（解决不可见字符显示问题）
    """
    print("=== 文本分割符号统计结果 ===")
    # 定义字符的可读名称映射（方便识别不可见字符）
    char_names = {
        ' ': '空格',
        '\n': '换行符',
        '\t': '制表符',
        '\r': '回车符'
    }
    
    # 按出现次数降序排序
    sorted_items = sorted(separator_count.items(), key=lambda x: x[1], reverse=True)
    
    for char, count in sorted_items:
        # 获取字符的可读名称
        char_name = char_names.get(char, char)
        print(f"分割符号【{char_name}】: 出现 {count} 次")

# ------------------- 测试使用 -------------------
if __name__ == "__main__":
    # 示例文本（你可以替换成自己的目标文本）
    test_text = """
    正则表达式，是处理字符串的强大工具！它可以快速匹配、查找、替换文本中的特定模式。
    比如：用正则提取所有数字，或者分割复杂的文本\t这都很方便。
    学习正则，告别数据清洗的烦恼～
    """
    
    # 分析分割符号
    stats = analyze_separators(test_text)
    
    # 打印统计结果
    print_separator_stats(stats)