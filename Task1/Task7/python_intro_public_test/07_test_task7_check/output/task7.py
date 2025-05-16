def find_modified_max_argmax(l, f):
    r = [f(x) for x in l if type(x) == int]
    return (t := max(r), r.index(t)) if r else ()
