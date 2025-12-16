# 학습을 위한 pdf 데이터 자동 다운로드

import asyncio
from playwright.async_api import async_playwright
import re

async def download_pdfs(keyword="", max_downloads=5, page_num = 1):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="C:\Program Files\Google\Chrome\Application/chrome.exe", headless=False)  # headless=True로 설정하면 창 안뜸
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        # 접속
        await page.goto("https://www.open.go.kr/othicInfo/infoList/orginlInfoList.do")
        await page.wait_for_load_state("networkidle")

        # 검색어 입력
        try:
            await page.evaluate('''({ start, end }) => {
                const $ = window.jQuery;
                if (!$) throw new Error('jQuery not loaded');
                $('input[name="startDate"]').datepicker('setDate', start);
                $('input[name="endDate"]').datepicker('setDate', end);
            }''', {"start": "2024-01-01", "end": "2024-01-31"})
        except Exception as e:
            print("❌ 날짜 설정 실패:", e)

        await page.fill("#kwd", '')
        await page.wait_for_timeout(1000)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(1500)
        await page.wait_for_load_state("networkidle")

        count = 0
        #page_num = 115
        await page.evaluate(f"goPageInfo({page_num})")
        await page.wait_for_load_state("networkidle")

        while count < max_downloads:
            print(f"📄 페이지 {page_num} 처리 중...")
            
            await page.wait_for_timeout(2000)
            items = await page.query_selector_all("div.info_list li")
            if not items:
                print("❌ 항목 없음, 종료")
                break
        
            # 검색 결과 항목들 가져오기
            items = await page.query_selector_all("div.info_list li")
    
            for i in range(len(items)):
                if count >= max_downloads:
                    break

                selector = f"div.info_list li:nth-child({i+1}) a"
                await page.wait_for_load_state("networkidle")
                href = await page.eval_on_selector(selector, "el => el.getAttribute('href')")
                await page.wait_for_load_state("networkidle")

                # 정규식으로 goDetail 인자 파싱
                match = re.search(r"goDetail\('([^']+)','([^']+)','([^']+)','([^']+)'\)", href)

                if not match:
                    print(f"❌ {i+1}번째 항목: goDetail 파싱 실패")
                    continue

                doc_id, code, type_code, idx = match.groups()

                # 상세페이지로 이동
                await page.evaluate(f"goDetail('{doc_id}', '{code}', '{type_code}', '{idx}')")
                await page.wait_for_load_state("networkidle")

                try:
                    await page.wait_for_selector("a.btn_type05.down", timeout=2000)
                    await page.wait_for_load_state("networkidle")
                except Exception as e:
                    print("❌ 다운로드 실패_2:", e)
                    
                    try:
                        await page.go_back()
                    except Exception as e:
                        print("❌ 다운로드 실패_1:", e)
                        await browser.close()  # 브라우저 먼저 닫기
                        await download_restart(page_num)
                        return
                    
                    continue

                # wonmunStep1 인자 추출
                onclick = await page.eval_on_selector("a.btn_type05.down", "el => el.getAttribute('onclick')")
                #match = re.search(r"wonmunStep1\('([^']+)','([^']+)','([^']+)'\)", onclick)
                #match = re.search(r"wonmunStep1\('([^']+)',\s*'([^']+)',\s*'([^']+)'\)", onclick)
                #match = re.search(r"wonmunStep1\('([^']+)',\s*'([^']+)',\s*'([^']+)'(?:,\s*'([^']+)')?\)", onclick)
                #match = re.search(r"wonmunStep1\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\)",onclick)
                match = re.search(r"wonmunStep1\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*(?:,\s*'([^']+)')?\s*\)",onclick)
                await page.wait_for_load_state("networkidle")

                if not match:
                    print(f"❌ {i+1}번째 항목: wonmunStep1 파싱 실패")
                    await browser.close()
                    await download_restart(page_num)
                    #await page.go_back()
                    #await page.wait_for_load_state("networkidle")
                    return

                #wonmun_id, filename, public_flag = match.groups()

                groups = match.groups()

                if len(groups) == 3:
                    wonmun_id, filename, public_flag = groups
                elif len(groups) == 4:
                    wonmun_id, filename, public_flag, extra = groups
                else:
                    print("❌ 예상 외의 인자 개수")
                    ...
                    return

                number = str(page_num) + "_" + str(i + 1)

                # PDF 다운로드
                print(f"📥 {i+1}. {filename} 다운로드 중...")

                try:
                    async with page.expect_download(timeout=10000) as download_info:
                        
                        # await page.evaluate(f"""() => {{
                        #     javascript:wonmunStep1('{wonmun_id}', '{filename}', '{public_flag}');
                        # }}""")

                        await page.evaluate(f"""() => {{
                            javascript:wonmunStep1('{wonmun_id}', '{filename}', '{public_flag}', '{extra if len(groups)==4 else '0'}');
                        }}""")

                    download = await download_info.value
                    #await download.save_as(r"I:\\doc_data\\" + "202401\\" + number + "_" + filename)
                    await download.save_as(r"E:\pdf\\" + number + "_" + filename)
                    print(f"✅ 저장 완료: {filename}")
                    count += 1
                except Exception as e:
                    count += 1                    
                    print(f"❌ 다운로드 실패: {e}")

                # 목록으로 돌아가기
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(1000)
                try:
                    await page.go_back()
                    await page.wait_for_load_state("networkidle")
                except Exception as e:
                    await page.evaluate(f"goPageInfo({next_page_num})")    
                    await page.wait_for_load_state("networkidle")          

            # 다음 페이지 존재 여부 확인
            next_page_num = page_num + 1
            await page.wait_for_function("typeof goPageInfo === 'function'", timeout=2000)

            try:
                await page.evaluate(f"goPageInfo({next_page_num})")
            except Exception as e:
                print(f"🚫{next_page_num} 다음 페이지 이동 실패: {e}")                
                page_num = next_page_num
                await page.click("a[title='다음으로']")  # 다음 페이지 이동
                await page.wait_for_load_state("networkidle")
                continue

            await page.wait_for_load_state("networkidle")
            page_num = next_page_num

async def download_restart(page_num):
    
    await download_pdfs('', 300000, page_num)


asyncio.run(download_pdfs("", max_downloads=300000, page_num = 100))

