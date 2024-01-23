def derive(f, x, h=0.0001):
    """
    Computes the derivative of the function f at point x using central difference.
    f: function to differentiate
    x: point at which to evaluate the derivative
    h: small step size
    """
    return (f(x + h) - f(x - h)) / (2 * h)