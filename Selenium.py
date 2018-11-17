from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import  *
from time import sleep

driver_init = False
markets = {
    "NORTH AMERICA": ["UNITED STATES", "CANADA", "CRYPTOCURRENCY"],
    "LATIN AMERICA": ["MEXICO", "ARGENTINA", "CHILE"],
    "EUROPE": ["LONDON", "FRANKFURT", "AMSTERDAM", "PARIS", "BRUSSELS", "LISBON", "SWITZERLAND", "IRELAND", "MILAN",
               "BUDAPEST", "ATHENS", "BERLIN", "HAMBURG", "HANOVER", "DUSSELDORF", "MUNICH", "DENMARK", "FINLAND",
               "SWEDEN", "SPAIN", "NORWAY"],
    "ASIA": ["MUMBAI", "HONGKONG", "KUALALUMPUR", "NEWZEALAND", "SHANGHAI", "SYDNEY", "IS", "JAKARTA", "BANGKOK",
             "SHENZEN", "TOKYO"]
}


class Bot:

    def __init__(self):
        self.driver = webdriver.Chrome('C:/Users/jikiw/Documents/chromedriver')
        self.driver.get("https://www.stocktrak.com/")
        self.__state = "initializing"
        self.__login()
        self.__remove_tip()

    def __login(self):
        login_button = self.driver.find_element_by_id('login-btn')
        login_button.click()

        username_field = self.driver.find_element_by_id('tbLoginUserName')
        password_field = self.driver.find_element_by_id('Password')

        username_field.send_keys('xxxxxx')
        password_field.send_keys('xxxxxx', Keys.ENTER)
        sleep(2)
        self.__state = "logged in"

    def close(self):
        if not driver_init: raise ValueError('Please init driver before trying anything')
        self.driver.close()
        self.__state = "closed"

    def __remove_tip(self):
        try:
            self.driver.find_element_by_id('btn-remindlater').click()
            self.__state = "dashboard"
        except NoSuchElementException:
            print('There was no tip')
            self.__state = "dashboard"

    def __trade(self):
        self.driver.get("https://www.stocktrak.com/trading/equities")
        self.__state = "trading"

    def __fill_order(self, symbol, quantity, action_type="Cover", order_type="Market", limit_stop_price="",
                   order_term="good-til-day"):
        if order_type != "Market" and limit_stop_price == "":
            raise ValueError("Please fill in a limit/stop price when the order type is different from market")
        switch = Select(self.driver.find_element_by_id('ddlOrderSide'))

        if action_type == "Buy":
            switch.select_by_value("1")
        elif action_type == "Sell":
            switch.select_by_value("2")
        elif action_type == "Short":
            switch.select_by_value("3")
        elif action_type == "Cover":
            switch.select_by_value("4")

        symbol_field = self.driver.find_element_by_id("tbSymbol")
        symbol_field.send_keys(symbol)

        quantity_field = self.driver.find_element_by_id("tbQuantity")
        quantity_field.send_keys(str(quantity))

        order_switch = Select(self.driver.find_element_by_id("ddlOrderType"))

        if order_type == "Market":
            order_switch.select_by_value("1")
        elif order_type == "Limit":
            order_switch.select_by_value("2")
        elif order_type == "Stop":
            order_switch.select_by_value("3")
        elif order_type == "Trailing Stop $":
            order_switch.select_by_value("4")
        elif order_type == "Trailing Stop %":
            order_switch.select_by_value("5")

        if order_type != "Market":
            limit_stop_price_field = self.driver.find_element_by_id("tbPrice")
            limit_stop_price_field.send_keys(str(limit_stop_price))

            order_term_switch = Select(self.driver.find_element_by_id("ddlOrderExpiration"))

            if order_term == "Good-til-Day":
                order_term_switch.select_by_value("1")
            elif order_term == "Good-til-Cancel":
                order_term_switch.select_by_value("2")
            elif order_term == "Good-til-Date":
                order_term_switch.select_by_value("3")

        self.__state = "order filled"

    def __switch_region(self, region):
        if region not in ["LATIN AMERICA", "NORTH AMERICA", "EUROPE", "ASIA"]: raise ValueError('Region specified is not accepted.')
        switch = Select(self.driver.find_element_by_id('ddlRegion'))
        print(switch)
        if region == "NORTH AMERICA":
            switch.select_by_value("NorthAmerica")
        elif region == "LATIN AMERICA":
            switch.select_by_value("LatinAmerica")
        elif region == "EUROPE":
            switch.select_by_value("Europe")
        elif region == "ASIA":
            switch.select_by_value("Asia")

        self.__state = "order fill"

    def __preview(self):
        sleep(2)
        preview_button = self.driver.find_element_by_id('btnPreviewOrder')
        preview_button .click()

        self.__state = "preview"

    def __confirm(self, message):
        sleep(2)
        notes = self.driver.find_element_by_id("trade-notes")
        notes.send_keys(message)
        preview_button = self.driver.find_element_by_id('btnPlaceOrder')
        preview_button .click()
        self.__state = "confirmed"

    def __show_dashboard(self):
        if self.__state != "dashboard":
            self.driver.get("https://www.stocktrak.com/dashboard")
            self.__state = "dashboard"

    def __read_money(self):
        """Must be on the dashboard to get the amount of money"""
        if self.__state != "dashboard":
            return

        snapshot_container = self.driver.find_element_by_id("snapshot-container")
        table = snapshot_container.find_element_by_css_selector('*')
        tbody = table.find_element_by_css_selector('*')
        trs = tbody.find_elements_by_xpath(".//*")
        print(trs)

        elements = list(filter(lambda elem: len(elem) == 2, list(map(lambda tr: list(map(lambda td: td.text, tr.find_elements_by_xpath(".//*"))), trs))))
        elem = {elem[0]: elem[1] for elem in elements}

        return elem

    def place_order(self, symbol, quantity, action_type="Cover", order_type="Market", limit_stop_price="",
                    order_term="good-til-day", message="Automatically generated trade"):

        if self.__state == "closed":
            self.__init__()

        if self.__state in ["confirmed", "preview", "order filled", "dashboard"]:
            self.__trade()

        self.__switch_region("NORTH AMERICA")
        self.__fill_order(symbol, quantity, action_type, order_type, limit_stop_price, order_term)
        self.__preview()
        self.__confirm(message)

    def get_money_remaining(self):
        self.__show_dashboard()
        self.__read_money()



bot = Bot()
# bot.place_order("FB", 1, action_type="Buy", order_type="Market")
bot.get_money_remaining()