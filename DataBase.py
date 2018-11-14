import Database_credentials as dc
import mysql.connector
import pandas as pd


def get_positions(tickers):
    assert(type(tickers) is list)
    tickers = tuple(tickers)

    conn = mysql.connector.connect(host=dc.host, user=dc.user, password=dc.password, database=dc.database)
    cursor = conn.cursor()
    cursor.execute("""SELECT Ticker, LongPosition, Invested, ShortPosition, Provisioned FROM Positions WHERE Ticker IN {}""".format(tickers))
    rows = cursor.fetchall()
    ret_val = pd.DataFrame(rows, columns=['Ticker', 'LongPosition', 'Invested', 'ShortPosition', 'Provisioned']).set_index('Ticker')
    cursor.close()
    conn.close()

    return ret_val


def update_position(ticker, long, invested, short, provisioned):
    conn = mysql.connector.connect(host=dc.host, user=dc.user, password=dc.password, database=dc.database)
    cursor = conn.cursor()
    cursor.execute("""UPDATE Positions SET  LongPosition = '{}', Invested = '{}', ShortPosition = '{}', Provisioned = '{}' WHERE Ticker = '{}'""".format(long, invested, short, provisioned, ticker))
    conn.commit()
    cursor.close()
    conn.close()


def open_position(ticker, position, entry_date, entry_price, entry_money):
    conn = mysql.connector.connect(host=dc.host, user=dc.user, password=dc.password, database=dc.database)
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO PositionsLog (Ticker, Position, EntryDate, EntryPrice, EntryMoney) VALUES ('{}', '{}', '{}', '{}', '{}')""".format(ticker, position, entry_date, entry_price, entry_money))
    conn.commit()
    cursor.close()
    conn.close()


def close_position(id, exit_date, exit_price, exit_money, profit):
    conn = mysql.connector.connect(host=dc.host, user=dc.user, password=dc.password, database=dc.database)
    cursor = conn.cursor()
    cursor.execute("""UPDATE PositionsLog SET  ExitDate = '{}', ExitPrice = '{}', ExitMoney = '{}', Profit = '{}' WHERE id = '{}'""".format(exit_date, exit_price, exit_money, profit, id))
    conn.commit()


    cursor.close()
    conn.close()
    return id


def reset(tickers):
    assert (type(tickers) is list)
    tickers = tuple(tickers)

    conn = mysql.connector.connect(host=dc.host, user=dc.user, password=dc.password, database=dc.database)
    cursor = conn.cursor()
    cursor.execute("""UPDATE Positions SET  LongPosition = '0', Invested = '0', ShortPosition = '0', Provisioned = '0' WHERE Ticker IN {}""".format(tickers))
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    tickers = ['AABA', 'AAL', 'AAOI', 'AAPL', 'AAXJ', 'AAXN', 'ABEO', 'ABIL', 'ABIO', 'ABMD', 'ACAD', 'ACET', 'ACGL', 'ACHC', 'ACHN', 'ACIA', 'ACIW', 'ACLS', 'ACOR', 'ACRX', 'ACST', 'ACWI', 'ACWX', 'ADAP', 'ADBE', 'ADI', 'ADMP', 'ADMS', 'ADOM', 'ADP', 'ADRO', 'ADSK', 'ADTN', 'ADVM', 'ADXS', 'AEIS', 'AERI', 'AEZS', 'AFIN', 'AFMD', 'AFSI', 'AGEN', 'AGIO', 'AGNC', 'AGRX', 'AIMT', 'AINV', 'AIPT', 'AKAM', 'AKAO', 'AKBA', 'AKCA', 'AKER', 'AKRX', 'ALDR', 'ALGN', 'ALKS', 'ALNY', 'ALRM', 'ALTR', 'ALXN', 'AMAG', 'AMAT', 'AMBA', 'AMBC', 'AMCX', 'AMD', 'AMED', 'AMGN', 'AMKR', 'AMRH', 'AMRN', 'AMRS', 'AMR', 'AMTD', 'AMZN', 'ANGI', 'ANSS', 'AOBC', 'APPN', 'APPS', 'APRI', 'APTI', 'APTO', 'AQMS', 'ARAY', 'ARCC', 'ARCE', 'ARDX', 'AREX', 'ARLP', 'ARNA', 'ARQL', 'ARRS', 'ARRY', 'ARWR', 'ASML', 'ASNA', 'ASRT', 'ATHN', 'ATHX', 'ATIS', 'ATRA', 'ATRS', 'ATSG', 'ATVI', 'AUPH', 'AVEO', 'AVGO', 'AVGR', 'AVID', 'AVT', 'AVXL', 'AXAS', 'AXTI', 'AY', 'AZPN', 'BABY', 'BBBY', 'BBOX', 'BCOR', 'BCRX', 'BDSI', 'BEAT', 'BECN', 'BGCP', 'BGFV', 'BHF', 'BIDU', 'BIIB', 'BILI', 'BIOS', 'BJRI', 'BKCC', 'BKNG', 'BLCM', 'BLDP', 'BLDR', 'BLMN', 'BLNK', 'BLPH', 'BLRX', 'BLUE', 'BL', 'BMCH', 'BMRN', 'BNDX', 'BND', 'BOTZ', 'BPFH', 'BPMC', 'BPOP', 'BPR', 'BPY', 'BRKL', 'BRKR', 'BRKS', 'BRY', 'BZUN', 'CAKE', 'CALA', 'CALM', 'CAMP', 'CAPR', 'CARA', 'CARB', 'CARG', 'CAR', 'CASA', 'CASI', 'CASY', 'CATB', 'CATM', 'CATY', 'CBAY', 'CBIO', 'CBLK', 'CBOE', 'CBSH', 'CCIH', 'CDEV', 'CDK', 'CDMO', 'CDNA', 'CDNS', 'CDW', 'CELG', 'CENX', 'CERC', 'CERN', 'CERS', 'CFFN', 'CFMS', 'CGNX', 'CG', 'CHKP', 'CHRS', 'CHRW', 'CHTR', 'CINF', 'CLBK', 'CLDX', 'CLMT', 'CLNE', 'CLPS', 'CLSN', 'CLVS', 'CMCSA', 'CME', 'CNAT', 'CNET', 'CNSL', 'COHR', 'COLL', 'COMM', 'CONE', 'CONN', 'CORT', 'COST', 'COUP', 'COWN', 'CPLP', 'CPRT', 'CPRX', 'CPST', 'CRAY', 'CRBP', 'CREE', 'CRNT', 'CRON', 'CROX', 'CRSP', 'CRTO', 'CRUS', 'CRZO', 'CSCO', 'CSFL', 'CSIQ', 'CSOD', 'CSQ', 'CSX', 'CTAS', 'CTRE', 'CTRL', 'CTRP', 'CTSH', 'CTXS', 'CUR', 'CVBF', 'CVLT', 'CYBR', 'CYCC', 'CYHHZ', 'CYTK', 'CYTR', 'CY', 'CZR', 'DBX', 'DCIX', 'DELT', 'DERM', 'DFFN', 'DISCA', 'DISCK', 'DISH', 'DLTR', 'DMPI', 'DNKN', 'DNLI', 'DOCU', 'DOMO', 'DOX', 'DRNA', 'DRRX', 'DRYS', 'DVAX', 'DVY', 'DXCM', 'EARS', 'EA', 'EBAY', 'ECHO', 'ECYT', 'EDGE', 'EDIT', 'EEFT', 'EFII', 'EGLE', 'EGOV', 'EIGI', 'EKSO', 'ELGX', 'EMB', 'ENDP', 'ENPH', 'ENTG', 'ENT', 'EOLS', 'EPAY', 'EPZM', 'EQIX', 'ERIC', 'ERII', 'ERI', 'ESEA', 'ESIO', 'ESPR', 'ESRX', 'ETFC', 'ETSY', 'EUFN', 'EVOP', 'EWBC', 'EXAS', 'EXEL', 'EXPD', 'EXPE', 'EXTR', 'EYEG', 'EYES', 'EYE', 'EYPT', 'EZPW', 'FANG', 'FAST', 'FATE', 'FB', 'FCEL', 'FEYE', 'FFBC', 'FFIV', 'FGEN', 'FHB', 'FISV', 'FITB', 'FIVE', 'FIVN', 'FLEX', 'FLIR', 'FLNT', 'FLXN', 'FMBI', 'FNJN', 'FNKO', 'FNSR', 'FOCS', 'FOLD', 'FORM', 'FOSL', 'FOXA', 'FOX', 'FPRX', 'FRAN', 'FRED', 'FRGI', 'FRTA', 'FSCT', 'FSLR', 'FTNT', 'FTR', 'FTSM', 'FULT', 'FV', 'FWONK', 'GALT', 'GBCI', 'GBT', 'GDS', 'GERN', 'GEVO', 'GGAL', 'GIII', 'GILD', 'GLIBA', 'GLNG', 'GLPI', 'GLUU', 'GLYC', 'GMLP', 'GNCA', 'GNMK', 'GNTX', 'GOGO', 'GOLD', 'GOOGL', 'GOOG', 'GOV', 'GPOR', 'GPRE', 'GPRO', 'GRFS', 'GRMN', 'GRPN', 'GSKY', 'GSM', 'GTXI', 'GT', 'GWPH', 'HABT', 'HAIN', 'HALO', 'HAS', 'HA', 'HBAN', 'HCSG', 'HDP', 'HDSN', 'HDS', 'HEAR', 'HIBB', 'HIIQ', 'HIMX', 'HLIT', 'HMHC', 'HMNY', 'HMSY', 'HOLI', 'HOLX', 'HOMB', 'HOPE', 'HPT', 'HQY', 'HRTX', 'HSGX', 'HSIC', 'HTBX', 'HTGM', 'HTHT', 'HTLD', 'HWC', 'HZNP', 'IAC', 'IART', 'IBB', 'IBKC', 'ICHR', 'ICON', 'ICPT', 'IDTI', 'IDXG', 'IDXX', 'IEF', 'IEI', 'IGIB', 'IGSB', 'IIVI', 'ILMN', 'IMDZ', 'IMGN', 'IMMR', 'IMMU', 'IMPV', 'INCY', 'INFI', 'INFN', 'INFO', 'INOV', 'INO', 'INSM', 'INSY', 'INTC', 'INTU', 'INVA', 'IONS', 'IOVA', 'IPGP', 'IQ', 'IRBT', 'IRDM', 'IRWD', 'ISBC', 'ISRG', 'ITCI', 'IUSG', 'IUSV', 'IXUS', 'IZEA', 'JACK', 'JAGX', 'JAZZ', 'JBHT', 'JBLU', 'JCOM', 'JD', 'JKHY', 'KBWB', 'KERX', 'KEYW', 'KHC', 'KLAC', 'KLIC', 'KLXE', 'KNDI', 'KOPN', 'KPTI', 'KRNY', 'KTOS', 'KTWO', 'LAMR', 'LASR', 'LAUR', 'LBRDK', 'LBTYA', 'LBTYK', 'LECO', 'LGCY', 'LGIH', 'LILAK', 'LITE', 'LIVN', 'LJPC', 'LKQ', 'LLNW', 'LNTH', 'LOGI', 'LOGM', 'LOXO', 'LPLA', 'LPNT', 'LPSN', 'LRCX', 'LSCC', 'LSXMA', 'LSXMK', 'LTBR', 'LTRPA', 'LULU', 'LXRX', 'LX', 'MANH', 'MARA', 'MARK', 'MAR', 'MASI', 'MAT', 'MBB', 'MBFI', 'MBRX', 'MB', 'MCHI', 'MCHP', 'MDB', 'MDCA', 'MDCO', 'MDLZ', 'MDRX', 'MDSO', 'MDXG', 'MEET', 'MELI', 'MEOH', 'MGI', 'MHLD', 'MIDD', 'MIK', 'MITK', 'MITL', 'MKSI', 'MLCO', 'MLHR', 'MLNX', 'MMSI', 'MMYT', 'MNGA', 'MNKD', 'MNRO', 'MNST', 'MNTA', 'MOBL', 'MOMO', 'MOSY', 'MRCY', 'MRNS', 'MRTX', 'MRVL', 'MSFT', 'MTBC', 'MTCH', 'MTSI', 'MU', 'MVIS', 'MXIM', 'MYGN', 'MYL', 'MYSZ', 'MZOR', 'NAKD', 'NATI', 'NAVI', 'NBEV', 'NBIX', 'NCMI', 'NDAQ', 'NEOS', 'NEO', 'NEPT', 'NETE', 'NFLX', 'NIHD', 'NKTR', 'NLNK', 'NMIH', 'NMRK', 'NTAP', 'NTCT', 'NTES', 'NTGR', 'NTLA', 'NTNX', 'NTRI', 'NTRS', 'NUAN', 'NUVA', 'NVAX', 'NVCR', 'NVDA', 'NWBI', 'NWSA', 'NWS', 'NXPI', 'NXST', 'NXTD', 'NXTM', 'NYMT', 'OCLR', 'OCSL', 'OCUL', 'ODFL', 'ODP', 'OHGI', 'OHRP', 'OKTA', 'OLED', 'OLLI', 'OMER', 'ONB', 'ONCE', 'ONCS', 'ONVO', 'ON', 'OPHT', 'OPK', 'OPRA', 'OPTT', 'ORBC', 'ORBK', 'ORLY', 'OSTK', 'OSUR', 'OTEX', 'OTIC', 'OTIV', 'OVAS', 'OZK', 'PAAS', 'PACB', 'PACW', 'PAYX', 'PBCT', 'PBYI', 'PCAR', 'PCH', 'PCRX', 'PDBC', 'PDCE', 'PDCO', 'PDD', 'PDLI', 'PEGA', 'PEGI', 'PEIX', 'PENN', 'PEP', 'PETQ', 'PETS', 'PETX', 'PFF', 'PFG', 'PFPT', 'PGNX', 'PINC', 'PIRS', 'PIXY', 'PI', 'PLAB', 'PLAY', 'PLCE', 'PLUG', 'PNFP', 'PODD', 'PPBI', 'PPC', 'PRAA', 'PRAH', 'PRGS', 'PRPO', 'PRTA', 'PRTK', 'PSEC', 'PSTI', 'PS', 'PTCT', 'PTC', 'PTEN', 'PTIE', 'PTI', 'PTLA', 'PULM', 'PXLW', 'PYPL', 'PZZA', 'QCOM', 'QGEN', 'QIWI', 'QNST', 'QQQ', 'QRTEA', 'QRVO', 'QTNA', 'QTT', 'RARE', 'RBBN', 'RCII', 'RCM', 'RCON', 'RDFN', 'RDUS', 'REGI', 'REGN', 'RGLD', 'RGNX', 'RGSE', 'RIGL', 'RING', 'RIOT', 'RMBS', 'ROBO', 'ROIC', 'ROKU', 'ROST', 'RP', 'RRGB', 'RRR', 'RSLS', 'RSYS', 'RTRX', 'RUN', 'RWLK', 'RYAAY', 'SABR', 'SAFM', 'SAGE', 'SANM', 'SBAC', 'SBGI', 'SBLK', 'SBNY', 'SBRA', 'SBUX', 'SCHN', 'SCYX', 'SCZ', 'SEDG', 'SEIC', 'SESN', 'SFIX', 'SFLY', 'SFM', 'SFNC', 'SGEN', 'SGH', 'SGMO', 'SGMS', 'SGYP', 'SHIP', 'SHOO', 'SHPG', 'SHV', 'SHY', 'SIMO', 'SINA', 'SINO', 'SIRI', 'SIR', 'SIVB', 'SLGN', 'SLM', 'SLS', 'SMPL', 'SMRT', 'SMTC', 'SNBR', 'SNCR', 'SND', 'SNES', 'SNH', 'SNPS', 'SOHU', 'SOLO', 'SONC', 'SONO', 'SORL', 'SOXX', 'SPEX', 'SPHS', 'SPI', 'SPLK', 'SPPI', 'SPWH', 'SPWR', 'SQQQ', 'SRAX', 'SRCL', 'SREV', 'SRNE', 'SRPT', 'SRRA', 'SSC', 'SSNC', 'SSP', 'SSRM', 'SSYS', 'STAY', 'STKL', 'STLD', 'STMP', 'STX', 'SUPN', 'SVMK', 'SWIR', 'SWKS', 'SYMC', 'SYNA', 'SYNH', 'TACO', 'TANH', 'TCBI', 'TEAM', 'TECD', 'TELL', 'TENB', 'TERP', 'TGTX', 'TILE', 'TIVO', 'TLGT', 'TLRY', 'TLT', 'TMUS', 'TNDM', 'TNXP', 'TOPS', 'TQQQ', 'TRIP', 'TRMB', 'TROV', 'TROW', 'TRUE', 'TRVG', 'TRVN', 'TSCO', 'TSEM', 'TSG', 'TSLA', 'TSRO', 'TTD', 'TTMI', 'TTNP', 'TTOO', 'TTPH', 'TTS', 'TTWO', 'TUES', 'TUR', 'TVIX', 'TVTY', 'TWLVR', 'TWNK', 'TWOU', 'TXMD', 'TXN', 'TXRH', 'UAL', 'UBNT', 'UBSI', 'UCBI', 'UCTT', 'UFPI', 'UGLD', 'ULTA', 'UMPQ', 'UNFI', 'UNIT', 'UPL', 'URBN', 'USAT', 'USCR', 'USLV', 'UTHR', 'UXIN', 'VCEL', 'VCIT', 'VCSH', 'VC', 'VECO', 'VEON', 'VERI', 'VIAB', 'VIAV', 'VIOT', 'VIRT', 'VKTX', 'VMBS', 'VNDA', 'VNET', 'VNOM', 'VNQI', 'VOD', 'VRAY', 'VRA', 'VREX', 'VRNS', 'VRNT', 'VRSK', 'VRSN', 'VRTX', 'VSAT', 'VSTM', 'VTGN', 'VTIP', 'VTL', 'VTVT', 'VUZI', 'VXUS', 'WAFD', 'WATT', 'WBA', 'WB', 'WDAY', 'WDC', 'WEN', 'WERN', 'WETF', 'WIFI', 'WING', 'WIN', 'WIX', 'WLTW', 'WMGI', 'WPRT', 'WSC', 'WTFC', 'WWR', 'WYNN', 'XEL', 'XGTI', 'XLNX', 'XLRN', 'XNET', 'XOG', 'XON', 'XPER', 'XRAY', 'XSPA', 'YNDX', 'YRCW', 'YY', 'ZAGG', 'ZBRA', 'ZGNX', 'ZG', 'ZION', 'ZIOP', 'ZNGA', 'ZN', 'ZS', 'ZUMZ', 'ZYNE', 'Z']
    reset(tickers)
    print(get_positions(tickers))