import ast
import operator as op


ops = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg
}


MATH_ERROR = "Error: Unsupported expression, only support operations +,-,*,/,**"


def calculator(expr: str) -> str:
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        else:
            raise TypeError(MATH_ERROR)

    try:
        node = ast.parse(expr, mode='eval').body
        return str(_eval(node))
    except Exception:
        return MATH_ERROR
