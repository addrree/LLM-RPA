from typing import Any


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
        selector = args["selector"]
        locator = page.locator(selector)
        match_count = await locator.count()

        index = args.get("index")
        if index is not None:
            target_locator = locator.nth(index)
            args["_executor_note"] = (
                f'Selector "{selector}" matched {match_count} elements; used index={index}.'
            )
        else:
            target_locator = locator.first
            if match_count > 1:
                args["_executor_note"] = (
                    f'Selector "{selector}" matched {match_count} elements; used first element.'
                )

        return (await target_locator.inner_text()).strip()

    async def extract_html(self, page, args):
        selector = args["selector"]
        locator = page.locator(selector)
        match_count = await locator.count()

        index = args.get("index")
        if index is not None:
            target_locator = locator.nth(index)
            args["_executor_note"] = (
                f'Selector "{selector}" matched {match_count} elements; used index={index}.'
            )
        else:
            target_locator = locator.first
            if match_count > 1:
                args["_executor_note"] = (
                    f'Selector "{selector}" matched {match_count} elements; used first element.'
                )

        return (await target_locator.inner_html()).strip()

    async def extract_items(self, page, args):
        container_selector = args["container_selector"]
        limit = int(args["limit"])
        fields = args["fields"]

        containers = page.locator(container_selector)
        count = min(await containers.count(), limit)
        items = []

        for idx in range(count):
            container = containers.nth(idx)
            item = {}
            for field_name, rule in fields.items():
                item[field_name] = await self._extract_field_value(container, field_name, rule)
            items.append(item)

        args["_executor_note"] = f'Extracted {len(items)} item(s) via "{container_selector}" (limit={limit}).'
        return items

    async def _extract_field_value(self, container, field_name: str, rule: Any):
        selector = None
        attr = None

        if isinstance(rule, str):
            if rule.endswith(".href"):
                selector = rule[: -len(".href")]
                attr = "href"
            else:
                selector = rule
                if rule == "a" and "link" in field_name.lower():
                    attr = "href"
        elif isinstance(rule, dict):
            selector = rule.get("selector")
            attr = rule.get("attr")

        if not selector:
            return None

        locator = container.locator(selector).first
        if attr:
            value = await locator.get_attribute(attr)
            return (value or "").strip()

        return (await locator.inner_text()).strip()

    async def screenshot(self, page, args):
        path = args["path"]
        await page.screenshot(path=path)
        return path
