
import yaml, hashlib, json, datetime, math, os, csv, sys, numpy as np
import rg_core, skyrmion_v2 as sk2

def sha256_file(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()

def main():
    with open('params.yaml') as f:
        P = yaml.safe_load(f)

    # --- RG evolution ---
    alpha = 1 / P['alpha_inv']
    e_base = (4*math.pi*alpha) ** 0.5
    g2_0 = e_base / (P['sin2_in'] ** 0.5)
    g1_0 = (5/3)**0.5 * e_base / ((1-P['sin2_in'])**0.5)
    g3_0 = (4*math.pi*P['alpha_s_L']) ** 0.5
    yt_0 = (2**0.5) * P['m_top'] / 246.22
    lam_0 = (125.0**2)/(2*246.22**2)
    y0 = [g1_0, g2_0, g3_0, yt_0, lam_0]

    coupl = rg_core.evolve(P['Lambda_TeV']*1e3, 91.1876, y0, decouple=P['decouple'])
    sin2_MZ = coupl['sin2W']

    # --- Skyrmion Hessian ---
    lam_min = sk2.hessian_min(P['Lambda_TeV'], 
                              lambda_Sk=P['lambda_Sk'],
                              kappa_BI=P['kappa_BI'], 
                              lambda_col=P['lambda_col'])

    # --- store results ---
    result = {
        'timestamp': datetime.datetime.utcnow().isoformat()+'Z',
        'input_sha256': sha256_file('params.yaml'),
        'sin2_MZ': sin2_MZ,
        'lambda_min': lam_min
    }
    with open('result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
