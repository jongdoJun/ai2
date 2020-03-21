from crime_map.controller import Controller
from crime_map.folium_test import FoliumTest
if __name__ == '__main__':
    def print_menu():
        print('0. Exit')
        print('1. MODELING')
        print('2. CRIME MAP')
        print('3. 미국 실업률 지도')
        return input('메뉴 선택 \n')
    app = Controller()
    while 1:
        menu = print_menu()
        if menu == '1':
            app.police_pos()
            app.cctv_pop()
            app.police_norm()
        if menu == '2':
            app.crime_map()
        if menu == '3':
            t = FoliumTest()
            t.show_map()
        elif menu == '0':
            break