"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Tag,
  FolderOpen,
  CheckCircle2,
  FlaskConical,
  QrCode,
  ScrollText,
  Users,
  BookOpen,
  Sparkles,
  ShoppingCart,
  Calendar,
  ChevronLeft,
  ChevronRight,
  Bell,
  Search,
  Settings,
  Menu,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/promotions", label: "Promotions", icon: Tag },
  { href: "/campaigns", label: "Campaigns", icon: FolderOpen },
  { href: "/approvals", label: "Approvals", icon: CheckCircle2, badge: 2 },
  { href: "/simulator", label: "Simulator", icon: FlaskConical },
  { href: "/scenarios", label: "Scenarios", icon: BookOpen },
  { href: "/codes", label: "Codes", icon: QrCode },
  { href: "/calendar", label: "Calendar", icon: Calendar },
  { href: "/catalog", label: "Catalog", icon: ShoppingCart },
  { href: "/ai-assistant", label: "AI Assistant", icon: Sparkles },
  { href: "/audit", label: "Audit Trail", icon: ScrollText },
  { href: "/users", label: "Users", icon: Users },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          flex flex-col bg-white border-r border-gray-200 shadow-sm
          transition-all duration-300 ease-in-out
          ${collapsed ? "w-[72px]" : "w-64"}
          ${mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
        `}
      >
        {/* Logo */}
        <div className={`flex items-center h-16 border-b border-gray-200 ${collapsed ? "justify-center px-2" : "px-4"}`}>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-bass-gold flex items-center justify-center flex-shrink-0">
              <span className="text-white font-extrabold text-sm">BP</span>
            </div>
            {!collapsed && (
              <div className="flex flex-col">
                <span className="text-sm font-bold text-gray-900 leading-tight">PromoEngine</span>
                <span className="text-[10px] text-gray-500 font-medium">Bass Pro Shops</span>
              </div>
            )}
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-3 px-2.5 space-y-0.5">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className={`sidebar-link ${active ? "sidebar-link-active" : "sidebar-link-inactive"} ${collapsed ? "justify-center px-2" : ""}`}
                title={collapsed ? item.label : undefined}
              >
                <Icon size={20} className="flex-shrink-0" />
                {!collapsed && (
                  <span className="flex-1">{item.label}</span>
                )}
                {!collapsed && item.badge && (
                  <span className="bg-red-500 text-white text-[10px] font-bold rounded-full w-5 h-5 flex items-center justify-center">
                    {item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Collapse toggle */}
        <div className="border-t border-gray-200 p-2.5 hidden lg:block">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="sidebar-link sidebar-link-inactive w-full justify-center"
          >
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
            {!collapsed && <span className="flex-1">Collapse</span>}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-4 lg:px-6 flex-shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMobileOpen(true)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100"
            >
              <Menu size={20} />
            </button>
            <div className="relative hidden sm:block">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search promotions, campaigns, codes..."
                className="pl-9 pr-4 py-2 text-sm bg-gray-50 border border-gray-200 rounded-lg w-80 focus:outline-none focus:ring-2 focus:ring-bass-gold/30 focus:border-bass-gold"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline-flex items-center px-2.5 py-1 rounded-md bg-amber-50 text-amber-700 text-xs font-semibold border border-amber-200">
              TEST
            </span>
            <button className="relative p-2 rounded-lg hover:bg-gray-100 text-gray-500">
              <Bell size={20} />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full" />
            </button>
            <button className="p-2 rounded-lg hover:bg-gray-100 text-gray-500">
              <Settings size={20} />
            </button>
            <div className="ml-2 flex items-center gap-2.5 pl-3 border-l border-gray-200">
              <div className="w-8 h-8 rounded-full bg-bass-green flex items-center justify-center">
                <span className="text-white text-xs font-bold">AR</span>
              </div>
              <div className="hidden md:flex flex-col">
                <span className="text-sm font-semibold text-gray-900 leading-tight">Alan Reed</span>
                <span className="text-[11px] text-gray-500">Super Admin</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
