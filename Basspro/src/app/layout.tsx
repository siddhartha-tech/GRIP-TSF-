import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PromoEngine | Bass Pro Shops",
  description: "Promotion Engine Admin Console for Bass Pro Shops",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans">{children}</body>
    </html>
  );
}
