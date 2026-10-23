import os
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from google import genai
from google.genai import types


# =========================================================
# Configuration
# =========================================================

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Medical AI Analyzer"],
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)


# =========================================================
# Medical image analysis prompt
# =========================================================

MEDICAL_REPORT_PROMPT = """
You are an AI medical imaging assistant, not a physician.

Analyze the supplied image conservatively.
The image may be a pelvic ultrasound, an obstetric ultrasound,
or another medical image.

Write the report in professional Arabic.

Use these sections:

1. نوع الصورة:
   State the modality and anatomy only if reasonably identifiable.

2. جودة الصورة وحدود التقييم:
   Describe visibility, cropping, image artifacts, and limitations.
   If the image is unclear or insufficient, say so explicitly.

3. الملاحظات المرئية:
   Describe only findings that are genuinely visible in the image.
   For pelvic ultrasound, mention ovarian or follicular appearance
   only if the anatomy is visible and can reasonably be assessed.
   Do not invent measurements, follicle counts, ovarian volume,
   blood flow, fetal heartbeat, or other findings not demonstrated
   by the supplied image.

4. القياسات:
   Report a measurement only if it is explicitly visible and legible
   in the image. Otherwise write "غير متاح من الصورة".

5. الخلاصة:
   Give a cautious summary of the visible image.
   Do not claim a confirmed diagnosis from an image alone.
   If the image is inadequate, state that no reliable conclusion
   can be made from it.

6. الخطوات التالية:
   Suggest appropriate clinician review or additional assessment
   when indicated. Do not prescribe medication or supplements.

Important safeguards:
- Never invent a finding to fill an empty report section.
- Distinguish visible observations from uncertainty.
- Do not claim that PCOS is confirmed or excluded based on this
  image alone.
- Do not claim to have examined the patient.
- Symptoms and age are context, not proof of an imaging finding.
- This is an AI-generated draft requiring qualified clinical review.
"""


# =========================================================
# Gemini analysis function
# =========================================================

def analyze_image_with_gemini(
    image_bytes: bytes,
    mime_type: str,
    age: Optional[str],
    symptoms: Optional[str],
) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "مفتاح GEMINI_API_KEY غير مضبوط. "
                "أضفه إلى متغيرات البيئة في إعدادات الخادم."
            ),
        )

    client = genai.Client(api_key=api_key)

    patient_context = (
        f"العمر المدخل: {age.strip() if age and age.strip() else 'غير محدد'}\n"
        f"الأعراض المدخلة: "
        f"{symptoms.strip() if symptoms and symptoms.strip() else 'غير محددة'}"
    )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            MEDICAL_REPORT_PROMPT,
            patient_context,
            image_part,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1800,
        ),
    )

    report_text = response.text

    if not report_text or not report_text.strip():
        raise RuntimeError("The model returned an empty response.")

    return report_text.strip()


# =========================================================
# API endpoint
# POST /api/analyze-ultrasound
# =========================================================

@router.post("/analyze-ultrasound")
async def analyze_ultrasound_image(
    age: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    try:
        # Validate content type.
        mime_type = (image.content_type or "").lower().strip()

        if mime_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    "نوع الصورة غير مدعوم. "
                    "الأنواع المقبولة: JPG وPNG وWEBP."
                ),
            )

        # Read a bounded amount of data to prevent oversized uploads.
        image_bytes = await image.read(MAX_IMAGE_SIZE + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="الملف المرفق فارغ.",
            )

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="حجم الصورة يتجاوز الحد المسموح به وهو 10 MB.",
            )

        # Run the synchronous SDK call outside the async event loop.
        analysis_text = await run_in_threadpool(
            analyze_image_with_gemini,
            image_bytes,
            mime_type,
            age,
            symptoms,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        report = (
            "تقرير تحليل صورة طبية بمساعدة الذكاء الاصطناعي\n"
            f"تاريخ إنشاء التقرير: {timestamp}\n"
            f"العمر المدخل: {age.strip() if age and age.strip() else 'غير محدد'}\n"
            f"الأعراض المدخلة: "
            f"{symptoms.strip() if symptoms and symptoms.strip() else 'غير محددة'}\n\n"
            "ملاحظات النموذج:\n"
            f"{analysis_text}\n\n"
            "تنبيه: هذا تقرير آلي أولي، وليس تشخيصًا طبيًا نهائيًا. "
            "تجب مراجعته من مختص مؤهل."
        )

        return {
            "status": "success",
            "message": "تم إنشاء مسودة التقرير.",
            "report": report,
            "analysis": analysis_text,
            "timestamp": timestamp,
            "model": GEMINI_MODEL,
            "is_diagnostic": False,
            "requires_clinician_review": True,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Ultrasound image analysis failed.")

        raise HTTPException(
            status_code=502,
            detail=(
                "تعذر إكمال تحليل الصورة. "
                "تحقق من مفتاح Gemini واتصال الخادم وتوافق النموذج، "
                "ثم حاول مرة أخرى."
            ),
        )

    finally:
        await image.close()
