def check(x: str, file: str):
    f = open(file, 'w')
    words = {}
    for i in x.lower().split(" "):
        if i not in words.keys():
            words[i] = 1
        else:
            words[i] += 1
    a = sorted(words.items())
    for i in a:
        print(*i, file=f)
