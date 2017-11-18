import image_open as io
import simple_preprocess as sp
import feature_extractor as fe


def run():
    train, valid, test = io.read_MNIST()
    br = []
    holes = []
    lines = []
    for digit in range(0,10):
        digit_imgs, digit_indices = io.MNIST_sort(train, digit,25000)
        #digit_proc_thres1, digit_proc_thres2 = sp.run_image_preprocess_MNIST(digit_imgs)
        digit_proc = sp.run_image_preprocess_MNIST(digit_imgs)
        #br_digit, holes_digit = fe.features_MNIST(digit_proc_thres1, digit_proc_thres2)
        br_digit, holes_digit, lines_digit = fe.features_MNIST(digit_proc)
        br.append(br_digit)
        holes.append(holes_digit)
        lines.append(lines_digit)
    return br, holes, lines
