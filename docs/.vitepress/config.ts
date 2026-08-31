import { defineConfig } from "vitepress"

export default defineConfig({
  title: "frameworthy",
  description: "A lightweight testing library for dataframes and analytical transformations.",

  base: "/frameworthy/",

  themeConfig: {
    nav: [
      { text: "Guide", link: "/getting-started" },
    //   { text: "API", link: "/api/" }
    ],

    sidebar: [
      {
        text: "Guide",
        items: [
          { text: "Introduction", link: "/" },
          { text: "Getting Started", link: "/getting-started" }
        ]
      },
      {
        text: "Assertions",
        items: [
          { text: "Preserving Rows", link: "/assertions/preserves-rows" },
          { text: "Preserving Keys", link: "/assertions/preserves-key" }
        ]
      }
    ],

    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/joypauls/frameworthy"
      }
    ],

    search: {
      provider: "local"
    }
  }
})
