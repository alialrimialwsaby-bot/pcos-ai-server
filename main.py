from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

app = FastAPI(title="PCOS AI Clinical Intelligence Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    insulin: float = 0.0
    testosterone: float = 0.0
    tsh: float = 0.0
    prolactin: float = 0.0
    lh: float = 0.0
    fsh: float = 0.0
    dheas: float = 0.0
    shbg: float = 0.0
    hba1c: float = 0.0
    vitD: float = 0.0

class PCOSTheAIModel:
    def analyze_case(self, data: PatientInput) -> Dict[str, Any]:
        # محرك الاستدلال الذكي (AI Inference Logic)
        risk_scores = {
            "insulin_resistance": 0,
            "hyperandrogenism_ovarian": 0,
            "adrenal_stress": 0,
            "thyroid_prolactin": 0,
            "micronutrient_deficiency": 0
        }
        
        findings = []
        protocols = []

        # 1. تحليل محور مقاومة الإنسولين
        if data.insulin > 10.0 or data.hba1c >= 5.7:
            risk_scores["insulin_resistance"] += 2
            findings.append("• نموذج الذكاء الاصطناعي رصد مؤشرات واضحة لسيادة مقاومة الإنسولين الأيضية.")
            protocols.append("• بروتوكول الأيض: مكملات ميو-إينوسيتول وحمض الألفا ليبويك، مع حمية منخفضة المؤشر الجلايسيمي.")

        # 2. تحليل محور المبايض والأندروجينات
        if data.testosterone > 0.7 or (data.lh > 0 and data.lh > 10.0):
            risk_scores["hyperandrogenism_ovarian"] += 2
            findings.append("• نموذج الذكاء الاصطناعي رصد فرط أندروجين وارتفاع في هرمونات المبيض.")
            protocols.append("• بروتوكول التوازن الهرموني: شاي النعناع البلدي ومكملات الساو بالميتو لخفض الهرمونات الذكرية.")

        # 3. تحليل المحور الكظري (الإجهاد والتوتر)
        if data.dheas > 340.0:
            risk_scores["adrenal_stress"] += 2
            findings.append("• نموذج الذكاء الاصطناعي استنتج وجود إجهاد كظري (أدرينالي) ناتج عن التوتر العصبي.")
            protocols.append("• بروتوكول الغدة الكظرية: مغنيسيوم جلايسينات وأشواغاندا لخفض الكورتيزول.")

        # 4. الغدة الدرقية وهرمون الحليب
        if data.tsh > 4.0 or (0 < data.tsh < 0.4) or data.prolactin > 25.0:
            risk_scores["thyroid_prolactin"] += 2
            findings.append("• نموذج الذكاء الاصطناعي رصد تداخلاً في محاور الغدة الدرقية أو البرولاكتين.")
            protocols.append("• بروتوكول غددي: استشارة أخصائي غدد صماء للتأكد من استقرار TSH وهرمون الحليب.")

        # 5. نقص المغذيات وفيتامين د
        if 0 < data.vitD < 30.0:
            risk_scores["micronutrient_deficiency"] += 1
            findings.append("• نموذج الذكاء الاصطناعي حدد نقصاً مؤثراً في مخزون فيتامين د3 الداعم للمناعة والأيض.")
            protocols.append("• بروتوكول الدعم الحيوي: جرعة علاجية من فيتامين د3 مع كيه 2 (K2).")

        has_issue = len(findings) > 0

        # توليد تقرير الذكاء الاصطناعي الديناميكي
        if not has_issue:
            report = (
                "🤖 تقرير الذكاء الاصطناعي السريري:\n\n"
                "✅ بناءً على تحليل شبكة المؤشرات الحيوية، جميع القيم ضمن النطاق المستقر.\n\n"
                "• التوصية الذكية: الحفاظ على نمط حياة نشط، وتجنب السكريات البسيطة، والمتابعة الدورية."
            )
        else:
            report = (
                "🤖 **تقرير الذكاء الاصطناعي والتحليل الجذري:**\n\n" + 
                "\n".join(findings) + "\n\n" +
                "📋 **البروتوكول العلاجي الذكي المخصص:**\n\n" + 
                "\n".join(protocols) + "\n\n" +
                "🥗 **التوجيه الغذائي:** التركيز على البروتينات، الدهون الصحية، والخضروات الورقية تماماً.\n" +
                "🏃‍♀️ **التمارين:** المشي بعد الوجبات لمقاومة الإنسولين، أو اليوغا صباحاً لخفض الكورتيزول."
            )

        return {
            "hasResult": has_issue,
            "report": report,
            "ai_confidence_scores": risk_scores
        }

ai_model = PCOSTheAIModel()

@app.post("/api/analyze")
def analyze_patient(data: PatientInput):
    return ai_model.analyze_case(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
