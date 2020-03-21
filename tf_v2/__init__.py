from tf_v2.fashion_mnist import FashionMnist
from tf_v2.imdb import Imdb
from tf_v2.save_load import SaveLoad
if __name__ == '__main__':
    def print_menu():
        print('0. Exit')
        print('1. Fashion Mnist')
        print('2. IMDB')
        print('3. SaveLoad')
        return input('메뉴 선택 \n')

    while 1:
        menu = print_menu()
        if menu == '1':
            fm = FashionMnist()
            # fm.show_dataset()
            fm.creat_model()
            # fm.predict_image()
            # fm.subplot_test(fm.predict_image())
            print(fm.one_test(100))
        if menu == '2':
            imdb = Imdb()
            imdb.download_data()
            # print(imdb.create_sample())
        if menu == '3':
            sl = SaveLoad()
            sl.execute()
        elif menu == '0':
            break