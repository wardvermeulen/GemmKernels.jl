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
- [ ] abcd-aebf-dfce
- [ ] abcd-aebf-fdec
- [ ] abcd-aecf-bfde
- [ ] abcd-aecf-fbed
- [ ] abcd-aedf-bfce
- [ ] abcd-aedf-fbec
- [ ] abcd-aefb-fdce
- [ ] abcd-aefc-fbed
- [ ] abcd-eafb-fdec
- [ ] abcd-eafc-bfde
- [ ] abcd-eafd-fbec

# 5 x 4 x 5
- [o] abcdef-dega-gfbc
- [ ] abcdef-degb-gfac
- [ ] abcdef-degc-gfab
- [ ] abcdef-dfga-gebc
- [ ] abcdef-dfgb-geac
- [ ] abcdef-dfgc-geab
- [ ] abcdef-efga-gdbc
- [ ] abcdef-efgb-gdac
- [ ] abcdef-efgc-gdab
- [ ] abcdef-gdab-efgc
- [ ] abcdef-gdac-efgb
- [ ] abcdef-gdbc-efga
- [ ] abcdef-geab-dfgc
- [ ] abcdef-geac-dfgb
- [ ] abcdef-geac-dfgb
- [ ] abcdef-gebc-dfga
- [ ] abcdef-gfab-degc
- [ ] abcdef-gfac-degb
- [ ] abcdef-gfbc-dega