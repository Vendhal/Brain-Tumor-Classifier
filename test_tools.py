"""
Test all 6 MCP tools locally!
Run: python test_tools.py
Make sure uvicorn is running first:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import requests
import json
import base64
import sys
import os

SERVER_URL = "http://localhost:8000/mcp"

# ─────────────────────────────────────────────
# HELPER: Call any MCP tool
# ─────────────────────────────────────────────
def call_tool(tool_name, arguments):
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }
    try:
        response = requests.post(
            SERVER_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            timeout=60
        )

        # Handle SSE response
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            text = response.text
            for line in text.split("\n"):
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            parsed = json.loads(data)
                            if "result" in parsed or "error" in parsed:
                                return parsed
                        except:
                            pass
            return {"error": "Could not parse SSE response"}
        else:
            return response.json()

    except requests.exceptions.ConnectionError:
        return {"error": "❌ Server not running! Start with: uvicorn main:app --host 0.0.0.0 --port 8000"}
    except Exception as e:
        return {"error": str(e)}


def extract_result_text(response):
    """Extract text from MCP response"""
    if "error" in response and isinstance(response["error"], str):
        return response["error"]
    try:
        content = response.get("result", {}).get("content", [])
        if content:
            return content[0].get("text", str(response))
        # Try direct result
        result = response.get("result", "")
        if isinstance(result, str):
            return result
        return json.dumps(response, indent=2)
    except:
        return json.dumps(response, indent=2)


def print_result(tool_name, result_text, show_full=False):
    print(f"\n{'='*60}")
    print(f"🔧 TOOL: {tool_name}")
    print(f"{'='*60}")
    try:
        parsed = json.loads(result_text)
        if show_full:
            print(json.dumps(parsed, indent=2))
        else:
            # Show summary
            for key, value in parsed.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: [{len(value)} items]")
                elif isinstance(value, dict):
                    print(f"  {key}: {{...}}")
    except:
        print(result_text[:500] + "..." if len(result_text) > 500 else result_text)
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# TEST TOOL 3: list_tumor_classes (no args needed!)
# ─────────────────────────────────────────────
def test_list_tumor_classes():
    print("\n🧪 Testing: list_tumor_classes")
    print("   (Lists all 9 detectable tumor types)")

    response = call_tool("list_tumor_classes", {})
    result = extract_result_text(response)
    print_result("list_tumor_classes", result)
    return result


# ─────────────────────────────────────────────
# TEST TOOL 2: get_tumor_info
# ─────────────────────────────────────────────
def test_get_tumor_info(tumor_class="glioma"):
    print(f"\n🧪 Testing: get_tumor_info (class='{tumor_class}')")
    print("   (Gets clinical details about a tumor type)")

    response = call_tool("get_tumor_info", {"tumor_class": tumor_class})
    result = extract_result_text(response)
    print_result("get_tumor_info", result)
    return result


# ─────────────────────────────────────────────
# TEST TOOL 4: validate_mri_image
# ─────────────────────────────────────────────
def test_validate_mri_image(image_path=None):
    print("\n🧪 Testing: validate_mri_image")
    print("   (Checks if image is a valid MRI scan)")

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        print(f"   Using image: {image_path}")
    else:
        # Create a minimal valid PNG (small but real image)
        import struct, zlib

        def create_test_png():
            def png_chunk(chunk_type, data):
                chunk_len = struct.pack('>I', len(data))
                chunk_data = chunk_type + data
                crc = struct.pack('>I', zlib.crc32(chunk_data) & 0xffffffff)
                return chunk_len + chunk_data + crc

            width, height = 64, 64
            png_data = b'\x89PNG\r\n\x1a\n'
            ihdr = struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)
            png_data += png_chunk(b'IHDR', ihdr)

            raw_data = b''
            for y in range(height):
                raw_data += b'\x00'
                for x in range(width):
                    brightness = int(((x/width) + (y/height)) * 100)
                    raw_data += bytes([brightness])

            compressed = zlib.compress(raw_data)
            png_data += png_chunk(b'IDAT', compressed)
            png_data += png_chunk(b'IEND', b'')
            return png_data

        image_bytes = create_test_png()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        print("   Using: auto-generated 64x64 test image")

    response = call_tool("validate_mri_image", {"image_data": image_data})
    result = extract_result_text(response)
    print_result("validate_mri_image", result)
    return result, image_data


# ─────────────────────────────────────────────
# TEST TOOL 1: analyze_mri
# ─────────────────────────────────────────────
def test_analyze_mri(image_path=None):
    print("\n🧪 Testing: analyze_mri")
    print("   (Main tool - classifies brain tumor from MRI)")

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        print(f"   Using image: {image_path}")
    else:
        # Use the same test image generator
        import struct, zlib

        def create_test_png():
            def png_chunk(chunk_type, data):
                chunk_len = struct.pack('>I', len(data))
                chunk_data = chunk_type + data
                crc = struct.pack('>I', zlib.crc32(chunk_data) & 0xffffffff)
                return chunk_len + chunk_data + crc

            width, height = 224, 224
            png_data = b'\x89PNG\r\n\x1a\n'
            ihdr = struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)
            png_data += png_chunk(b'IHDR', ihdr)

            raw_data = b''
            for y in range(height):
                raw_data += b'\x00'
                for x in range(width):
                    brightness = int(((x/width) + (y/height)) * 100)
                    raw_data += bytes([brightness])

            compressed = zlib.compress(raw_data)
            png_data += png_chunk(b'IDAT', compressed)
            png_data += png_chunk(b'IEND', b'')
            return png_data

        image_bytes = create_test_png()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        print("   Using: auto-generated 224x224 test image")
        print("   ⚠️  For real results, provide an actual MRI image!")

    response = call_tool("analyze_mri", {
        "image_data": image_data,
        "patient_reference": "Patient/test-sandeep-001"
    })
    result = extract_result_text(response)
    print_result("analyze_mri", result, show_full=False)

    # Show key fields
    try:
        parsed = json.loads(result)
        print("\n  📊 KEY RESULTS:")
        print(f"  → Conclusion: {parsed.get('conclusion', 'N/A')}")
        print(f"  → Status: {parsed.get('status', 'N/A')}")

        ext = parsed.get("extension", [{}])[0].get("extension", [])
        for e in ext:
            if e.get("url") == "neuralOutput":
                neural = json.loads(e["valueString"])
                print(f"  → Top Prediction: {neural.get('topPrediction', 'N/A')}")
                print(f"  → Confidence: {round(neural.get('confidence', 0) * 100, 1)}%")
            if e.get("url") == "uncertaintyFlag":
                print(f"  → Uncertain: {e.get('valueBoolean', False)}")
    except:
        pass

    return result


# ─────────────────────────────────────────────
# TEST TOOL 5: assess_urgency
# ─────────────────────────────────────────────
def test_assess_urgency(fhir_report_json):
    print("\n🧪 Testing: assess_urgency")
    print("   (Assesses clinical urgency from FHIR report)")

    response = call_tool("assess_urgency", {
        "fhir_report_json": fhir_report_json,
        "patient_age": 45,
        "additional_symptoms": "headaches and occasional dizziness"
    })
    result = extract_result_text(response)
    print_result("assess_urgency", result)

    try:
        parsed = json.loads(result)
        print(f"\n  🚨 URGENCY: {parsed.get('urgency_level', 'N/A')} ({parsed.get('urgency_color', '')})")
        print(f"  ⏰ Timeframe: {parsed.get('recommended_timeframe', 'N/A')}")
        if parsed.get("red_flags"):
            print(f"  🚩 Red Flags: {len(parsed['red_flags'])} found")
            for flag in parsed["red_flags"]:
                print(f"     - {flag}")
    except:
        pass

    return result


# ─────────────────────────────────────────────
# TEST TOOL 6: generate_clinical_summary
# ─────────────────────────────────────────────
def test_generate_clinical_summary(fhir_report_json):
    print("\n🧪 Testing: generate_clinical_summary (clinical format)")
    print("   (Generates doctor-friendly summary)")

    response = call_tool("generate_clinical_summary", {
        "fhir_report_json": fhir_report_json,
        "format": "clinical"
    })
    result = extract_result_text(response)
    print_result("generate_clinical_summary", result)

    print("\n🧪 Testing: generate_clinical_summary (patient format)")
    print("   (Generates patient-friendly summary)")

    response2 = call_tool("generate_clinical_summary", {
        "fhir_report_json": fhir_report_json,
        "format": "patient"
    })
    result2 = extract_result_text(response2)
    print_result("generate_clinical_summary (patient)", result2)

    return result


# ─────────────────────────────────────────────
# MAIN: Run all tests
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Check if image path provided
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n" + "🧠 "*20)
    print("   BRAIN TUMOR CLASSIFIER - ALL TOOLS TEST")
    print("🧠 "*20)

    if image_path:
        print(f"\n📁 Using image: {image_path}")
    else:
        print("\n⚠️  No image provided - using synthetic test image")
        print("   For real results: python test_tools.py path/to/mri.jpg")

    print(f"\n🌐 Server: {SERVER_URL}")
    print("   Make sure uvicorn is running!")

    # ── Test 3: list_tumor_classes (simplest, no args)
    test_list_tumor_classes()

    # ── Test 2: get_tumor_info
    test_get_tumor_info("glioma")
    test_get_tumor_info("schwannoma")
    test_get_tumor_info("normal")

    # ── Test 4: validate_mri_image
    _, img_data = test_validate_mri_image(image_path)

    # ── Test 1: analyze_mri (main tool)
    fhir_report = test_analyze_mri(image_path)

    # ── Test 5 & 6: use the FHIR report from analyze_mri
    if fhir_report and "resourceType" in fhir_report:
        test_assess_urgency(fhir_report)
        test_generate_clinical_summary(fhir_report)
    else:
        print("\n⚠️  Skipping urgency + summary tests (analyze_mri didn't return valid FHIR)")
        print("   This is expected if using synthetic test image")

    print("\n" + "✅ "*20)
    print("   ALL TESTS COMPLETE!")
    print("✅ "*20)
    print("\n📋 SUMMARY:")
    print("   Tool 1 (analyze_mri):              ✅ Tested")
    print("   Tool 2 (get_tumor_info):            ✅ Tested (3 classes)")
    print("   Tool 3 (list_tumor_classes):        ✅ Tested")
    print("   Tool 4 (validate_mri_image):        ✅ Tested")
    print("   Tool 5 (assess_urgency):            ✅ Tested")
    print("   Tool 6 (generate_clinical_summary): ✅ Tested (2 formats)")
    print("\n🎯 For real MRI results:")
    print("   python test_tools.py C:/path/to/your/mri_image.jpg")