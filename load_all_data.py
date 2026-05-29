import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════
# LOAD ALL DATASETS
# ══════════════════════════════════════════════════════

def load_all_datasets():
    """Load all 9 files and return stats for AI context"""
    stats = {}

    # ── DATASET 1: Bundle Purchases (NEW_DATA) ─────────
    try:
        bp = pd.read_excel(
            'data/Bundle_Purchases_0501-0510.xlsx', header=4)
        bp.columns = ['Msisdn','BundleName','PlanId',
                      'PurchaseDate','SubscriptionCharge','Channel']
        bp['SubscriptionCharge'] = pd.to_numeric(
            bp['SubscriptionCharge'], errors='coerce').fillna(0)
        stats['bundle_new'] = {
            'total_purchases'   : len(bp),
            'unique_customers'  : bp['Msisdn'].nunique(),
            'total_revenue'     : round(bp['SubscriptionCharge'].sum(), 2),
            'avg_charge'        : round(bp['SubscriptionCharge'].mean(), 2),
            'date_range'        : 'May 1–9 2026',
            'top_bundles'       : bp['BundleName'].value_counts().head(8).to_dict(),
            'revenue_by_bundle' : bp.groupby('BundleName')['SubscriptionCharge']
                                    .sum().sort_values(ascending=False)
                                    .head(8).to_dict(),
            'channels'          : bp['Channel'].value_counts().to_dict(),
            'top_bundle'        : bp['BundleName'].value_counts().index[0],
            'top_bundle_count'  : int(bp['BundleName'].value_counts().iloc[0]),
        }
        print(f"✅ Bundle Purchases (NEW_DATA) loaded: {len(bp):,} records")
    except Exception as e:
        print(f"❌ Bundle Purchases error: {e}")
        stats['bundle_new'] = {}

    # ── DATASET 2: Addon Subscription (TELCO_DATA) ─────
    try:
        addon = pd.read_csv('data/ADDON_SUBSCRIPTION_REPORT.csv')
        addon['Subscription Charge'] = pd.to_numeric(
            addon['Subscription Charge'], errors='coerce').fillna(0)
        stats['addon'] = {
            'total_records'     : len(addon),
            'unique_customers'  : addon['Msisdn'].nunique(),
            'total_revenue'     : round(addon['Subscription Charge'].sum(), 2),
            'avg_charge'        : round(addon['Subscription Charge'].mean(), 2),
            'date_range'        : 'May 5 2026',
            'top_bundles'       : addon['Bundle Name'].value_counts().head(8).to_dict(),
            'revenue_by_bundle' : addon.groupby('Bundle Name')['Subscription Charge']
                                       .sum().sort_values(ascending=False)
                                       .head(8).to_dict(),
            'channels'          : addon['Channel'].value_counts().to_dict(),
            'top_bundle'        : addon['Bundle Name'].value_counts().index[0],
        }
        print(f"✅ Addon Subscription loaded: {len(addon):,} records")
    except Exception as e:
        print(f"❌ Addon Subscription error: {e}")
        stats['addon'] = {}

    # ── DATASET 3: November Data Usage ─────────────────
    try:
        nov = pd.read_excel(
            'data/MSISDN_DATAusage_November2025.xlsx', header=3)
        for col in nov.columns:
            if col != 'Number':
                nov[col] = pd.to_numeric(nov[col], errors='coerce').fillna(0)

        stats['nov_usage'] = {
            'total_customers'   : nov['Number'].nunique(),
            'total_usage_gb'    : round(nov['Usage (GB)'].sum(), 2),
            'avg_usage_gb'      : round(nov['Usage (GB)'].mean(), 2),
            'max_usage_gb'      : round(nov['Usage (GB)'].max(), 2),
            'month'             : 'November 2025',
            'facebook_gb'       : round(nov['101 Facebook'].sum()/(1024**3), 2),
            'youtube_gb'        : round(nov['107 Youtube'].sum()/(1024**3), 2),
            'tiktok_gb'         : round(nov['121 TikTok'].sum()/(1024**3), 2),
            'netflix_gb'        : round(nov['106 Netflix'].sum()/(1024**3), 2),
            'gaming_gb'         : round(nov['113 PUBG'].sum()/(1024**3), 2),
            'whatsapp_gb'       : round(nov['102 WhatsApp'].sum()/(1024**3), 2),
            'instagram_gb'      : round(nov['117 Instagram'].sum()/(1024**3), 2),
            'general_gb'        : round(nov['100 General Traffic'].sum()/(1024**3), 2),
        }
        print(f"✅ November Data Usage loaded: {nov['Number'].nunique():,} customers")
    except Exception as e:
        print(f"❌ November Usage error: {e}")
        stats['nov_usage'] = {}

    # ── DATASET 4: October Data Usage ──────────────────
    try:
        oct_df = pd.read_excel(
            'data/MSISDN_DATAusage_October2025.xlsx', header=3)
        for col in oct_df.columns:
            if col != 'Number':
                oct_df[col] = pd.to_numeric(oct_df[col], errors='coerce').fillna(0)

        oct_gb_col = 'Usage (GB)' if 'Usage (GB)' in oct_df.columns else None
        stats['oct_usage'] = {
            'total_customers'   : oct_df['Number'].nunique(),
            'total_usage_gb'    : round(oct_df[oct_gb_col].sum(), 2) if oct_gb_col else 0,
            'avg_usage_gb'      : round(oct_df[oct_gb_col].mean(), 2) if oct_gb_col else 0,
            'month'             : 'October 2025',
            'facebook_gb'       : round(oct_df['101 Facebook'].sum()/(1024**3), 2),
            'youtube_gb'        : round(oct_df['107 Youtube'].sum()/(1024**3), 2),
            'tiktok_gb'         : round(oct_df['121 TikTok'].sum()/(1024**3), 2),
            'gaming_gb'         : round(oct_df['113 PUBG'].sum()/(1024**3), 2),
        }
        print(f"✅ October Data Usage loaded: {oct_df['Number'].nunique():,} customers")
    except Exception as e:
        print(f"❌ October Usage error: {e}")
        stats['oct_usage'] = {}

    # ── DATASET 5: Postpaid November ───────────────────
    try:
        pp_nov = pd.read_excel(
            'data/MSISDN_Postpaid_November2025.xlsx', header=3)
        for col in pp_nov.columns:
            if col != 'Number':
                pp_nov[col] = pd.to_numeric(pp_nov[col], errors='coerce').fillna(0)
        stats['postpaid_nov'] = {
            'total_customers'   : pp_nov['Number'].nunique(),
            'total_usage_gb'    : round(pp_nov['Total (GB)'].sum(), 2),
            'avg_usage_gb'      : round(pp_nov['Total (GB)'].mean(), 2),
            'max_usage_gb'      : round(pp_nov['Total (GB)'].max(), 2),
            'month'             : 'November 2025',
        }
        print(f"✅ Postpaid November loaded: {pp_nov['Number'].nunique():,} customers")
    except Exception as e:
        print(f"❌ Postpaid November error: {e}")
        stats['postpaid_nov'] = {}

    # ── DATASET 6: Postpaid October ────────────────────
    try:
        pp_oct = pd.read_excel(
            'data/MSISDN_Postpaid_October2025.xlsx', header=3)
        for col in pp_oct.columns:
            if col != 'Number':
                pp_oct[col] = pd.to_numeric(pp_oct[col], errors='coerce').fillna(0)
        stats['postpaid_oct'] = {
            'total_customers'   : pp_oct['Number'].nunique(),
            'total_usage_gb'    : round(pp_oct['Total (GB)'].sum(), 2),
            'avg_usage_gb'      : round(pp_oct['Total (GB)'].mean(), 2),
            'month'             : 'October 2025',
        }
        print(f"✅ Postpaid October loaded: {pp_oct['Number'].nunique():,} customers")
    except Exception as e:
        print(f"❌ Postpaid October error: {e}")
        stats['postpaid_oct'] = {}

    # ── DATASET 7,8,9: Service Usage CSVs ──────────────
    try:
        s1 = pd.read_csv('data/16842521631.csv')
        s2 = pd.read_csv('data/16842727621.csv')
        s3 = pd.read_excel('data/16842588009.xlsx')
        su = pd.concat([s1, s2, s3], ignore_index=True)
        su['BENEFIT_GB'] = su['BENEFIT'] / (1024**3)
        su['USAGE_GB']   = su['USAGEVALUE'] / (1024**3)
        stats['service_usage'] = {
            'total_records'     : len(su),
            'unique_customers'  : su['EXTERNALSERVICEID'].nunique(),
            'active_records'    : int((su['STATUS']=='Active').sum()),
            'avg_benefit_gb'    : round(su['BENEFIT_GB'].mean(), 2),
            'avg_usage_gb'      : round(su['USAGE_GB'].mean(), 2),
            'max_usage_gb'      : round(su['USAGE_GB'].max(), 2),
        }
        print(f"✅ Service Usage files loaded: {len(su)} records")
    except Exception as e:
        print(f"❌ Service Usage error: {e}")
        stats['service_usage'] = {}

    return stats


if __name__ == "__main__":
    print("Loading all datasets...")
    print("-" * 50)
    stats = load_all_datasets()
    print("-" * 50)
    print("\n✅ All datasets loaded successfully!")
    print(f"\nSummary:")
    for key, val in stats.items():
        if isinstance(val, dict) and val:
            first_item = list(val.items())[0]
            print(f"  {key}: {first_item[0]}={first_item[1]}")