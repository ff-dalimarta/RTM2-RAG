#!/usr/bin/env python3
"""Literature Gap Matrix pipeline: local TF-IDF retrieval + optional Groq LLM generation.
No patient records; bundled source summaries are paraphrases of linked publication abstracts.
Usage: python rag_gap_pipeline.py --mode offline | groq
"""
import argparse,csv,json,os,re,math
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parent; DATA=BASE/'data'; OUT=BASE/'output'

def load_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def tokenize(s):return re.findall(r'[a-zA-Z0-9]+',s.lower())

def vectorize(documents):
    docs=[Counter(tokenize(s)) for s in documents]
    df=Counter(t for d in docs for t in d)
    n=len(docs)
    idf={t:math.log((n+1)/(v+1))+1 for t,v in df.items()}
    def transform(d):
        v={t:(1+math.log(k))*idf.get(t,0) for t,k in d.items() if t in idf}
        norm=math.sqrt(sum(x*x for x in v.values())) or 1
        return {t:x/norm for t,x in v.items()}
    return [transform(d) for d in docs], transform

def retrieve(papers,q,k=5):
    texts=[' '.join([p['title'],p['domain'],p['evidence_summary_paraphrase'],p['contribution'],p['tags']]) for p in papers]
    vectors,transform=vectorize(texts)
    query=transform(Counter(tokenize(q)))
    ranks=[]
    for p,v in zip(papers,vectors):
        score=sum(a*query.get(t,0) for t,a in v.items())
        ranks.append((score,p))
    return sorted(ranks,key=lambda x:x[0],reverse=True)[:k]

def generated_local(p):
    # Grounded extraction / deterministic generation, NOT an LLM. Never treat as validated gap finding.
    return {'existing_work':p['contribution'], 'reported_scope':p['evidence_summary_paraphrase'],
            'candidate_gap':p['research_gap_interpretation'],
            'gap_status':'ANALYST_INTERPRETATION_NOT_PROVEN_ABSENCE'}

def generated_groq(p):
    try:from groq import Groq
    except ImportError:raise SystemExit('pip install groq')
    if not os.getenv('GROQ_API_KEY'):raise SystemExit('Set GROQ_API_KEY securely')
    client=Groq(api_key=os.environ['GROQ_API_KEY'])
    msg=('Berdasarkan ringkasan sumber yang disediakan, keluarkan JSON dengan keys existing_work,reported_scope,candidate_gap,gap_status. '
         'Jangan klaim studi ini tidak pernah diteliti; tulis gap sebagai hipotesis yang perlu verifikasi full-text. '
         'Jangan menambah temuan, angka, atau referensi.\nSUMBER:\n'+json.dumps(p,ensure_ascii=False))
    result=client.chat.completions.create(model=os.getenv('GROQ_MODEL','openai/gpt-oss-20b'),temperature=0,
        messages=[{'role':'system','content':'Anda peneliti bidang farmasi dan kecerdasan buatan. Semua keluaran dalam bahasa Indonesia dan sesuai sumber.'},
                  {'role':'user','content':msg}])
    raw=result.choices[0].message.content or ''
    match=re.search(r'\{.*\}',raw,re.S)
    try:return json.loads(match.group(0) if match else raw)
    except (ValueError,AttributeError):return {'existing_work':p['contribution'],'reported_scope':p['evidence_summary_paraphrase'],
        'candidate_gap':p['research_gap_interpretation'],'gap_status':'LLM_JSON_FAILED_FALLBACK'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['offline','groq'],default='offline');ap.add_argument('--top-k',type=int,default=5);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    papers=load_csv(DATA/'papers.csv'); questions=load_csv(DATA/'research_questions.csv')
    matrix=[]; retrieval=[]
    for p in papers:
        g=generated_groq(p) if args.mode=='groq' else generated_local(p)
        matrix.append({**{x:p[x] for x in ['id','authors','year','title','venue','doi','url','domain']},**g})
    for q in questions:
        # Target tag is only part of the plain-text query. Retrieval itself uses TF-IDF similarities.
        ranked=retrieve(papers,q['question']+' '+q['target_tag'],args.top_k)
        for pos,(score,p) in enumerate(ranked,1):
            retrieval.append({'gap_id':q['gap_id'],'query':q['question'],'rank':pos,'paper_id':p['id'], 'score':round(score,5),'url':p['url']})
    with (OUT/'literature_gap_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=matrix[0].keys());w.writeheader();w.writerows(matrix)
    with (OUT/'retrieval_trace.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=retrieval[0].keys());w.writeheader();w.writerows(retrieval)
    with (OUT/'run_metadata.json').open('w',encoding='utf-8') as f:
        json.dump({'mode':args.mode,'retriever':'TF-IDF cosine (dependency-free local)','total_papers':len(papers),
          'top_k':args.top_k,'model':os.getenv('GROQ_MODEL') if args.mode=='groq' else None,
          'warning':'Hypothesized gaps based on annotated abstracts; verify against full texts before definitive novelty claim.'},f,indent=2)
    print('Saved:',OUT/'literature_gap_matrix.csv','and retrieval_trace.csv')
if __name__=='__main__':main()
