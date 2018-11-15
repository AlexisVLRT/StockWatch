from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from time import sleep

driver = webdriver.Chrome('C:/Users/jikiw/Documents/chromedriver')

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

def init():
    global driver_init
    driver_init = True
    driver.get("https://www.stocktrak.com/")


def login():
    if not driver_init: raise ValueError('Please init driver before trying anything')
    login_button = driver.find_element_by_id('login-btn')
    login_button.click()

    username_field = driver.find_element_by_id('tbLoginUserName')
    password_field = driver.find_element_by_id('Password')

    username_field.send_keys('xxxxxxx')
    password_field.send_keys('xxxxxxx', Keys.ENTER)
    sleep(2)

def close():
    if not driver_init: raise ValueError('Please init driver before trying anything')
    driver.close()


def remove_tip():
    driver.find_element_by_id('btn-remindlater').click()


def trade():
    driver.get("https://www.stocktrak.com/trading/equities")
    sleep(2)


def fill_order(symbol, quantity, action_type="Cover", order_type="Market", limit_stop_price="",
               order_term="good-til-day"):
    if order_type != "Market" and limit_stop_price == "":
        raise ValueError("Please fill in a limit/stop price when the order type is different from market")
    switch = Select(driver.find_element_by_id('ddlOrderSide'))

    if action_type == "Buy":
        switch.select_by_value("1")
    elif action_type == "Sell":
        switch.select_by_value("2")
    elif action_type == "Short":
        switch.select_by_value("3")
    elif action_type == "Cover":
        switch.select_by_value("4")

    symbol_field = driver.find_element_by_id("tbSymbol")
    symbol_field.send_keys(symbol)

    quantity_field = driver.find_element_by_id("tbQuantity")
    quantity_field.send_keys(str(quantity))

    order_switch = Select(driver.find_element_by_id("ddlOrderType"))

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
        limit_stop_price_field = driver.find_element_by_id("tbPrice")
        limit_stop_price_field.send_keys(str(limit_stop_price))

        order_term_switch = Select(driver.find_element_by_id("ddlOrderExpiration"))

        if order_term == "Good-til-Day":
            order_term_switch.select_by_value("1")
        elif order_term == "Good-til-Cancel":
            order_term_switch.select_by_value("2")
        elif order_term == "Good-til-Date":
            order_term_switch.select_by_value("3")


def switch_region(region):
    if region not in ["LATIN AMERICA", "NORTH AMERICA", "EUROPE", "ASIA"]: raise ValueError('Region specified is not accepted.')
    switch = Select(driver.find_element_by_id('ddlRegion'))
    print(switch)
    if region == "NORTH AMERICA":
        switch.select_by_value("NorthAmerica")
    elif region == "LATIN AMERICA":
        switch.select_by_value("LatinAmerica")
    elif region == "EUROPE":
        switch.select_by_value("Europe")
    elif region == "ASIA":
        switch.select_by_value("Asia")


def preview():
    sleep(2)
    preview_button = driver.find_element_by_id('btnPreviewOrder')
    preview_button .click()


def confirm():
    sleep(2)
    preview_button = driver.find_element_by_id('btnPlaceOrder')
    preview_button .click()


init()
login()
remove_tip()
trade()
switch_region("NORTH AMERICA")
fill_order("FB", 15, action_type="Buy", order_type="Market")
preview()
trade()
fill_order("FB", 15, action_type="Buy", order_type="Market")
