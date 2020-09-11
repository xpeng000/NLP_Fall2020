from typing import List

import regex

_RE_PUNC = regex.compile(r"^\p{p}$")
_EM_DASH = "-"

# note for grading TA:
# I couldn't import regex module, but I have activated conda and installed regex using pip The
# terminal gives me this when I tried to install regex again using pip: regex in
# /opt/anaconda3/lib/python3.8/site-packages (2020.6.8)
# I couldn't find an useful stackoverflow post on this. So I could use your help here! Thanks.


#1.Generating sentences
def gen_sentences(path):
    with open(path, encoding="utf8") as file:
        for line in file:
            #Here, I keep making copies of "line", is this a Shlemiel the painter problem?
            line = line.rstrip("\n")
            words: List[str] = line.split(" ")
            yield words


#2. Detokenizing
def detokenize(list_input):
    new_list = []
    start = True
    for i in range(len(list_input)):
        if _RE_PUNC.match(list_input[i]):
            #handle punctuations
            if list_input[i] == "\"" or list_input[i] == '"':
                if not start:
                    #add a space after
                    new_list.append(list_input[i]+" ")
                    start = True
                elif start and i != 0:
                    #add a space before
                    new_list.append(" "+list_input[i])
                    start = False
                else:
                    #just append to the list
                    new_list.append(list_input[i])
                start = False
            #handle hash, start and end punctuations
            elif list_input[i] == _EM_DASH or i == 0 or i == len(list_input)-1:
                new_list.append(list_input[i])
            #handle other cases by adding a space after
            else:
                new_list.append(list_input+" ")
        else:
            new_list.append(list_input[i])
    str_return = "".join(new_list)
    return str_return

#3. Sarcastic casting
def case_sarcastically(string):
    new_list = []
    count: int = 0
    for char in string:
        if char.lower() == char.upper():
            new_list.append(char)
        else:
            if count % 2 == 0:
                new_list.append(char)
            else:
                new_list.append(char.upper())
        count += 1
    str_sarc = "".join(new_list)
    return str_sarc


