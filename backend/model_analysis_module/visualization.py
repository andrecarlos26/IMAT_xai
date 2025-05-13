import base64

def encode_image(dot):
    """
    Takes a graphviz.Digraph and returns the base64 PNG image.
    """
    img_bytes = dot.pipe(format='png')
    return base64.b64encode(img_bytes).decode('utf-8')
