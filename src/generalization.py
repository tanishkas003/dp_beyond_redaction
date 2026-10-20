import re
from datetime import datetime
def generalize(value,entity_type):
    kind=(entity_type or "UNKNOWN").upper().replace(" ","_"); text=value.strip()
    if kind=="DATE":
        for fmt in ("%B %d, %Y","%B %d %Y","%d %B %Y","%Y-%m-%d"):
            try: return datetime.strptime(text,fmt).strftime("%B %Y")
            except ValueError: pass
        return "a recent date"
    if kind in {"AGE","YEARS_OLD"}:
        m=re.search(r"\d+",text)
        if m: low=int(m.group())//10*10; return f"{low}-{low+9} years old"
        return "an adult age range"
    if kind in {"FINANCIAL_AMOUNT","SALARY","AMOUNT"}:
        m=re.search(r"\d[\d,]*(?:\.\d+)?",text)
        if m:
            amount=float(m.group().replace(",","")); low=int(amount//100000*100000)
            return f"approximately {low:,}-{low+100000:,}"
        return "a financial amount"
    if kind in {"LOCATION","GPE","CITY"}: return "a regional location"
    if kind=="JOB_TITLE": return "a professional role"
    if kind in {"ORG","ORGANIZATION","COMPANY"}: return "a partner organization"
    raise ValueError(f"unsupported generalization type: {kind}")
