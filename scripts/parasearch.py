import math

def RSSI_A2A_OHuBig():
	Pt=20#dbm
	Gt=0
	Gr=0
	d=0.75#km
	f=5800#MHz
	ht=115#m
	hr=115#m
	C=0

	a_hr=3.2*(math.log10(11.754*hr))**2-4.97
	PL=69.55+26.16*math.log10(f)-13.82*math.log10(hr)-a_hr+(44.9-6.55*math.log10(ht))*math.log10(d)
	RSSI=Pt+Gt+Gr-PL
	return RSSI
def RSSI_A2G_OHuBig():
	Pt=20#dbm
	Gt=0
	Gr=20
	d=0.388#km
	f=5800#MHz
	ht=115#m
	hr=12#m

	a_hr=3.2*(math.log10(11.754*hr))**2-4.97
	PL=69.55+26.16*math.log10(f)-13.82*math.log10(hr)-a_hr+(44.9-6.55*math.log10(ht))*math.log10(d)
	RSSI=Pt+Gt+Gr-PL
	return RSSI
def RSSI_G2A_OHuBig():
	Pt=20#dbm
	Gt=0
	Gr=0
	d=0.388#km
	f=5800#MHz
	ht=12#m
	hr=115#m

	a_hr=3.2*(math.log10(11.754*hr))**2-4.97
	PL=69.55+26.16*math.log10(f)-13.82*math.log10(hr)-a_hr+(44.9-6.55*math.log10(ht))*math.log10(d)
	RSSI=Pt+Gt+Gr-PL
	return RSSI
print(RSSI_A2A_OHuBig())
print(RSSI_A2G_OHuBig())
print(RSSI_G2A_OHuBig())