# 3 x 3 x 2
- [x] abc-bda-dc (SVV)
- [x] abc-adb-dc (VVV) <span style="color:red">NOT PART OF TCCG</span>
- [x] abc-bda-cd (SVS) <span style="color:red">NOT PART OF TCCG</span>
- [x] abc-dca-bd

# 4 x 4 x 2
- [x] abcd-dbea-ec (SVV)
- [o] abcd-deca-be
- [o] abcd-ebad-ce

# 5 x 5 x 2
- [x] abcde-efbad-cf (SVS)
- [o] abcde-ecbfa-fd
- [o] abcde-efcad-bf

# 4 x 2 x 4
- [x] abcd-ea-ebcd (VSV) <span style="color:green">Vectorised store!</span> <span style="color:red">Bad performance still? A tensor also vectorised?</span> 
- [o] abcd-eb-aecd
- [o] abcd-ec-abed

# 2 x 2 x 2
- [x] ab-ac-cb

# 2 x 3 x 3
- [o] ab-acd-dbc
- [o] ab-cad-dcb

# 3 x 2 x 3
- [o] abc-ad-bdc

# 3 x 3 x 2
- [o] abc-acd-db
- [o] abc-adc-bd
- [o] abc-adc-db

# 3 x 4 x 3
- [x] abc-adec-ebd (SVV)
- [x] abc-adec-deb (SVV) <span style="color:red">NOT PART OF TCCG</span>
- [x] abc-adec-bde (SVS)  <span style="color:red">NOT PART OF TCCG</span>

# 4 x 4 x 4
- [o] abcd-aebf-dfce
- [o] abcd-aebf-fdec
- [o] abcd-aecf-bfde
- [o] abcd-aecf-fbed
- [o] abcd-aedf-bfce
- [o] abcd-aedf-fbec
- [o] abcd-aefb-fdce
- [o] abcd-aefc-fbed
- [o] abcd-eafb-fdec
- [o] abcd-eafc-bfde
- [o] abcd-eafd-fbec

# 5 x 4 x 5
- [o] abcdef-dega-gfbc
- [o] abcdef-degb-gfac
- [o] abcdef-degc-gfab
- [o] abcdef-dfga-gebc
- [o] abcdef-dfgb-geac
- [o] abcdef-dfgc-geab
- [o] abcdef-efga-gdbc
- [o] abcdef-efgb-gdac
- [o] abcdef-efgc-gdab
- [o] abcdef-gdab-efgc
- [o] abcdef-gdac-efgb
- [o] abcdef-gdbc-efga
- [o] abcdef-geab-dfgc
- [o] abcdef-geac-dfgb
- [o] abcdef-gebc-dfga
- [o] abcdef-gfab-degc
- [o] abcdef-gfac-degb
- [o] abcdef-gfbc-dega