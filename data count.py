import os

(_, knu, _) = next(os.walk('F:\CXR CT\COVID-19\KNU'))  # 343

for k in knu: # KNU0001
    (_, ct_cxr, cxr_fl) = next(os.walk(os.path.join('F:\CXR CT\COVID-19\KNU', k)))
    for fd1 in ct_cxr: # CT, CXR, CT ex
        (_, CT_ex, unkn_fl) = next(os.walk(os.path.join('F:\CXR CT\COVID-19\KNU', k, fd1)))
        if len(CT_ex) != 0:
            print(k + ' ' + fd1 + ' CT_ex: ' + str(len(CT_ex)))
            print('        ' + str(CT_ex))
        if len(unkn_fl) != 0:
            print(k + ' ' + fd1 + ' unkn_fl: ' + str(len(unkn_fl)))
            print('        ' + str(unkn_fl[:5]))
    if len(cxr_fl) != 0: # CXR files
        cxr_cnt = []
        for fl1 in cxr_fl:
            if fl1[3] == 'H':
                continue
            cxr_cnt.append(fl1)
        print(k + ' ' + ' cxr_fl: ' + str(len(cxr_cnt)))
        print('        ' + str(cxr_cnt))
