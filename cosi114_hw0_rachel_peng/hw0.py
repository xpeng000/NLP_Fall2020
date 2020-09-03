import regex
_RE_PUNC = regex.compile(r"^\p{p}$")
_EM_DSAH = "-"

#1. Generating sentences
#how to declare path variable?

def gen_sentences(path):
    with open(path, encoding="utf8") as file:
        for line in file:
            yield line.split(" ")
print(gen_sentences())


#2. Detokenizing
def detokenize(list_input):
    new_list = []
    for word in list_input:
        new_list.append(word)
    str_detoken = "".join(new_list)
    return str

#3. Sarcastic casting
def case_sarcastically(string):
    new_list = []
    count: int = 0
    for char in string:
        if char.lower() == char.upper():
            new_list.append(char)
        else:
            if count%2 == 0:
                new_list.append(char)
            else:
                new_list.append(char.upper())
        count += 1
    str_sarc = "".join(new_list)
    return str_sarc


