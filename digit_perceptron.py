import p3_utils
from load_data import DigitData

digit_data = DigitData("digitdata")


def cool_visualization():
    """
    I added this just to make sure that the loaded data is correct. But I ended up finding that if you stand far away
    from the monitor, you can actually see the digit. VERY COOL :P

    BTW, I do not recommend inverting the image. I'm just doing it for the visualization. Just get the datums
    directly from DigitData.digit_train_imgs and work with them. I'd recommend checking out the Datum class as well.
    :return:
    """
    for i, datum in enumerate(digit_data.digit_test_imgs):
        inverted_datum = p3_utils.array_invert(datum.get_pixels())
        for row in inverted_datum:
            print(row)
        if i > 4:
            break


if __name__ == '__main__':
    cool_visualization()
