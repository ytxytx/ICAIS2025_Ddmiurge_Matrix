#!/usr/bin/env python3
"""
Simple test script for selected API endpoints
"""
import base64
import httpx
import asyncio

BASE_URL = "http://localhost:8000"
PDF_FILE = "attention_is_all_you_need.pdf"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_ideation():
    """Test /ideation endpoint"""
    print_section("1. Testing /ideation")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/ideation",
            json={"query": "Generate research ideas for improving dexterous robotic manipulation, especially contact-rich tasks."}
        ) as response:
            print("Response:")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    content = line[6:]
                    if content == "[DONE]":
                        print("\n[Stream completed]")
                    else:
                        import json
                        try:
                            data = json.loads(content)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    print(delta["content"], end="", flush=True)
                        except:
                            pass




async def main():
    """Run only ideation and paper_review tests"""
    print("\n" + "=" * 80)
    print("  Science Arena Challenge - API Test Suite (Partial)")
    print("=" * 80)

    try:
        await test_ideation()

        print_section("Selected tests completed!")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
