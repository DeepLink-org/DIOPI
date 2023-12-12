# Copyright (c) 2023, DeepLink.
import re


class CodeTemplate(object):
    substitution_str = r'(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})'

    # older versions of Python have a bug where \w* does not work,
    # so we need to replace with the non-shortened version [a-zA-Z0-9_]*
    # https://bugs.python.org/issue18647

    substitution_str = substitution_str.replace(r'\w', r'[a-zA-Z0-9_]')

    subtitution = re.compile(substitution_str, re.MULTILINE)

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as f:
            return CodeTemplate(f.read())

    def __init__(self, pattern):
        self.pattern = pattern

    def substitute(self, env={}, **kwargs):
        def lookup(v):
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent, v):
            return ("".join([indent + line + "\n" for e in v
                    for line in str(e).splitlines()]).rstrip())

        def replace(match):
            indent = match.group(1)
            key = match.group(2)
            comma_before = ''
            comma_after = ''
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ', '
                    key = key[1:]
                if key[-1] == ',':
                    comma_after = ', '
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ', '.join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)
        return self.subtitution.sub(replace, self.pattern)
