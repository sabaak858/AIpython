def my_split(sentence, separator):
    result = []
    word = ""

    for ch in sentence:
        if ch == separator:
            result.append(word)
            word = ""
        else:
            word += ch
    result.append(word)
    return result


def my_join(items, separator):
    result = ""
    for i in range(len(items)):
        result += items[i]
        if i < len(items) - 1:
            result += separator
    return result
text = "apple,banana,cherry"
print(my_split(text, ","))
words = ["dog", "cat", "mouse"]
print(my_join(words, "-"))
