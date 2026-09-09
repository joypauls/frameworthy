import { defineConfig } from "vitepress"

export default defineConfig({
  title: "frameworthy",
  description: "Lightweight statistical validation library for dataframes and metric changes.",

  base: "/frameworthy/",

//   appearance: false,

  head: [
    ["link", { rel: "icon", type: "image/png", href: "/frameworthy/logo.png" }]
  ],

  themeConfig: {
    nav: [
      { text: "Docs", link: "/getting-started" },
    //   { text: "API", link: "/api/" }
    ],

    sidebar: [
      {
        text: "Guide",
        items: [
          // { text: "Home", link: "/" },
          { text: "Getting Started", link: "/getting-started" }
        ]
      },
      {
        text: "Internals",
        items: [
          { text: "Methodology", link: "/methodology" }
        ]
      },
      // {
      //   text: "API",
      //   items: [
      //     { text: "Preserving Rows", link: "/assertions/preserves-rows" },
      //     { text: "Preserving Keys", link: "/assertions/preserves-key" }
      //   ]
      // }
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
