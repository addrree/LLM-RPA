class ActionHandlers:
    async def open_url(self, page, args):
        await page.goto(args["url"])

    async def click(self, page, args):
        await page.click(args["selector"])

    async def type(self, page, args):
        await page.fill(args["selector"], args["text"])

    async def wait_for(self, page, args):
        await page.wait_for_selector(args["selector"])

    async def extract_text(self, page, args):
        return (await page.locator(args["selector"]).inner_text()).strip()

    async def extract_html(self, page, args):
        return await page.locator(args["selector"]).inner_html()

    async def screenshot(self, page, args):
        path = args["path"]
        await page.screenshot(path=path)
        return path