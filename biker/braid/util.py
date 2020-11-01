import re
import sys

def parse_api_link(url):

    #example intput: http://docs.oracle.com/javase/7/docs/api/java/text/NumberFormat.html#getInstance(java.util.Locale)
    #example output: (NumberFormat,getInstance)

    #print url

    url = url.split(".html")

    tokens = url[0].split("/")

    i = tokens.index('api')
    class_name = tokens[i+1]
    i = i+2
    while i < len(tokens):
        class_name = class_name+'.'+tokens[i]
        i = i+1

    #class_name = url[0].split("/")[-1]
    method_name = url[1]

    if method_name != '':
        method_name = method_name[1:]
        for i,ch in enumerate(method_name):
            if not method_name[i].isalpha():
                method_name = method_name[:i]
                break

    #print class_name,method_name

    return (class_name,method_name) #Note that class_name already contains package name


def normalize_dict(dic):

    min_value = sys.maxint
    max_value = -1

    for (k,v) in dic.items():
        min_value = min(min_value,v)
        max_value = max(max_value,v)

    for k in dic:
        dic[k] = (dic[k]-min_value+1)*1.0/(max_value-min_value+1)

